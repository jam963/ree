use crate::{config::Config, util::read_limited};
use anyhow::{Context, Result, bail, ensure};
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs},
    time::Duration,
};
use url::Url;

pub fn normalize(input: &str) -> Result<Url> {
    let mut url = Url::parse(input)?;
    ensure!(
        matches!(url.scheme(), "http" | "https"),
        "only HTTP(S) URLs are supported"
    );
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "URL credentials are not allowed"
    );
    ensure!(url.host_str().is_some(), "URL requires a host");
    url.set_fragment(None);
    Ok(url)
}
fn public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_multicast()
        || a == 0
        || a >= 240
        || (a == 100 && (64..=127).contains(&b))
        || (a == 198 && (b == 18 || b == 19))
        || (a == 192 && b == 0 && (c == 0 || c == 2)))
}
pub fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => public_v4(ip),
        IpAddr::V6(ip) => {
            if let Some(v4) = ip.to_ipv4_mapped() {
                return public_v4(v4);
            }
            let s = ip.segments();
            // Only native global unicast; disallow translation, 6to4, Teredo and documentation ranges.
            s[0] & 0xe000 == 0x2000
                && s[0] != 0x2002
                && !(s[0] == 0x2001 && (s[1] < 0x200 || s[1] == 0xdb8))
                && s[0] != 0x3fff
        }
    }
}
pub fn addresses(url: &Url, allow_private: bool) -> Result<Vec<SocketAddr>> {
    let host = url
        .host_str()
        .context("missing URL host")?
        .trim_matches(['[', ']']);
    let port = url.port_or_known_default().context("missing URL port")?;
    let addresses: Vec<_> = (host, port).to_socket_addrs()?.collect();
    ensure!(!addresses.is_empty(), "URL host has no addresses");
    ensure!(
        allow_private || addresses.iter().all(|a| public_ip(a.ip())),
        "private_network: URL resolves to a protected address"
    );
    Ok(addresses)
}
#[derive(Debug)]
pub struct Response {
    pub bytes: Vec<u8>,
    pub extension: String,
    pub final_url: String,
}
pub fn fetch(input: &str, config: &Config) -> Result<Response> {
    let mut url = normalize(input)?;
    for redirects in 0..=5 {
        let addresses = addresses(&url, config.allow_private_network)?;
        // Pin this request to the addresses checked above; disable ambient proxies
        // so DNS rebinding/proxy resolution cannot bypass private-network checks.
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .resolve_to_addrs(url.host_str().unwrap(), &addresses)
            .build()?;
        let response = client
            .get(url.clone())
            .header("User-Agent", concat!("ree/", env!("CARGO_PKG_VERSION")))
            .send()?;
        if response.status().is_redirection() {
            ensure!(redirects < 5, "redirect_limit: too many redirects");
            let next = url.join(
                response
                    .headers()
                    .get("location")
                    .context("redirect missing Location")?
                    .to_str()?,
            )?;
            url = normalize(next.as_str())?;
            continue;
        }
        let response = response.error_for_status()?;
        ensure!(
            response
                .content_length()
                .is_none_or(|n| n <= config.max_file_size),
            "size_limit: URL response is too large"
        );
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        let extension = if content_type.contains("html") {
            "html".into()
        } else if content_type.contains("pdf") {
            "pdf".into()
        } else {
            std::path::Path::new(url.path())
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("txt")
                .to_string()
        };
        return Ok(Response {
            bytes: read_limited(response, config.max_file_size)?,
            extension,
            final_url: url.to_string(),
        });
    }
    bail!("redirect limit exceeded")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identities() {
        assert_eq!(
            normalize("HTTPS://Example.com:443/a#b").unwrap().as_str(),
            "https://example.com/a"
        );
        assert!(normalize("file:///etc/passwd").is_err());
        assert!(normalize("https://user:pass@example.com").is_err());
    }
    #[test]
    fn protected_ranges() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "169.254.169.254",
            "100.64.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "2002:7f00:1::",
        ] {
            assert!(!public_ip(ip.parse().unwrap()), "{ip}");
        }
        assert!(public_ip("8.8.8.8".parse().unwrap()));
        assert!(public_ip("2606:4700:4700::1111".parse().unwrap()));
    }
}
