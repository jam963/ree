use ree::{cli::Options, config::Config, input::url};
use std::{
    io::{Read, Write},
    net::TcpListener,
};
fn serve(response: &'static str) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let worker = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buffer = [0; 4096];
        let _ = stream.read(&mut buffer);
        stream.write_all(response.as_bytes()).unwrap();
    });
    (format!("http://{address}/page"), worker)
}
#[test]
fn bounded_local_http_and_scheme_validation() {
    let mut config = Config::load(&Options::default()).unwrap();
    config.allow_private_network = true;
    let (address, worker) = serve(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 18\r\nConnection: close\r\n\r\n<main>Hello</main>",
    );
    let result = url::fetch(&address, &config).unwrap();
    assert_eq!(result.extension, "html");
    assert_eq!(result.bytes, b"<main>Hello</main>");
    worker.join().unwrap();
    let (address, worker) = serve(
        "HTTP/1.1 302 Found\r\nLocation: file:///etc/passwd\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );
    assert!(url::fetch(&address, &config).is_err());
    worker.join().unwrap();
    let (address, worker) =
        serve("HTTP/1.1 200 OK\r\nContent-Length: 999999999\r\nConnection: close\r\n\r\n");
    assert!(url::fetch(&address, &config).is_err());
    worker.join().unwrap();
}
#[test]
fn private_network_denied_before_connect() {
    let config = Config::load(&Options::default()).unwrap();
    assert!(
        url::fetch("http://127.0.0.1:1/", &config)
            .unwrap_err()
            .to_string()
            .contains("private_network")
    );
}
