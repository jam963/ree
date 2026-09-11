//! Bounded, single-runner Unix/stdio supervisor. CUDA lives only in its disposable
//! child; idle expiry/cancellation releases the process and its device context.
use super::protocol::{self, Frame, MAX_FRAME, MAX_RESPONSE, Operation};
use crate::{config::Config, error::AppError, events::Events};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    collections::{HashSet, VecDeque},
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, IntoRawFd, RawFd},
        unix::{
            fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt},
            net::{UnixListener, UnixStream},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const MAX_CLIENTS: usize = 8;
const QUEUE_PER_CLIENT: usize = 2;
const IO_TIMEOUT: Duration = Duration::from_secs(10);
static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}
struct Signals(Vec<(libc::c_int, libc::sigaction)>);
impl Signals {
    fn install() -> Result<Self> {
        STOP.store(false, Ordering::Relaxed);
        let mut guard = Self(vec![]);
        for signal in [libc::SIGINT, libc::SIGTERM] {
            // Only atomic state is touched by the handler; restore previous
            // dispositions on every return path, including partial setup.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                let mut old: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = stop as *const () as usize;
                libc::sigemptyset(&mut action.sa_mask);
                ensure!(
                    libc::sigaction(signal, &action, &mut old) == 0,
                    "install signal handler"
                );
                guard.0.push((signal, old));
            }
        }
        Ok(guard)
    }
}
impl Drop for Signals {
    fn drop(&mut self) {
        for (signal, old) in &self.0 {
            unsafe {
                libc::sigaction(*signal, old, std::ptr::null_mut());
            }
        }
    }
}
struct Nonblocking(RawFd, i32);
impl Nonblocking {
    fn new(fd: RawFd) -> Result<Self> {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        ensure!(
            flags >= 0 && unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == 0,
            "set nonblocking IO"
        );
        Ok(Self(fd, flags))
    }
}
impl Drop for Nonblocking {
    fn drop(&mut self) {
        unsafe {
            libc::fcntl(self.0, libc::F_SETFL, self.1);
        }
    }
}
fn ready(fd: RawFd, events: i16) -> bool {
    let mut p = libc::pollfd {
        fd,
        events,
        revents: 0,
    };
    unsafe {
        libc::poll(&mut p, 1, 0) > 0 && p.revents & (events | libc::POLLHUP | libc::POLLERR) != 0
    }
}
fn read_fd(fd: RawFd, bytes: &mut [u8]) -> std::io::Result<usize> {
    let n = unsafe { libc::read(fd, bytes.as_mut_ptr().cast(), bytes.len()) };
    if n < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}
fn write_fd(fd: RawFd, bytes: &[u8]) -> std::io::Result<usize> {
    let n = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
    if n < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}
fn transient(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
    )
}

struct Endpoint {
    listener: UnixListener,
    path: PathBuf,
    identity: (u64, u64),
    _lock: File,
}
impl Endpoint {
    fn bind(path: &Path) -> Result<Self> {
        ensure!(path.is_absolute(), "socket path must be absolute");
        let parent = path
            .parent()
            .context("socket needs a parent directory")?
            .canonicalize()?;
        let meta = parent.metadata()?;
        let uid = unsafe { libc::geteuid() };
        ensure!(
            meta.uid() == uid && meta.mode() & 0o077 == 0,
            "socket parent must be user-owned and mode 0700"
        );
        let path = parent.join(path.file_name().context("socket needs a filename")?);
        let mut name = path.as_os_str().to_owned();
        name.push(".lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(name)?;
        let meta = lock.metadata()?;
        ensure!(
            meta.is_file() && meta.uid() == uid && meta.nlink() == 1 && meta.mode() & 0o077 == 0,
            "unsafe socket lock file"
        );
        fs2::FileExt::try_lock_exclusive(&lock).context("worker already owns this socket")?;
        if let Ok(meta) = std::fs::symlink_metadata(&path) {
            ensure!(
                meta.file_type().is_socket() && meta.uid() == uid,
                "refusing to replace a non-socket or foreign endpoint"
            );
            match UnixStream::connect(&path) {
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => {
                    std::fs::remove_file(&path)?
                }
                _ => anyhow::bail!("socket already has a listener; refusing to replace it"),
            }
        }
        let listener = UnixListener::bind(&path)?;
        let meta = path.symlink_metadata()?;
        let endpoint = Self {
            listener,
            path,
            identity: (meta.dev(), meta.ino()),
            _lock: lock,
        };
        std::fs::set_permissions(&endpoint.path, std::fs::Permissions::from_mode(0o600))?;
        endpoint.listener.set_nonblocking(true)?;
        Ok(endpoint)
    }
}
impl Drop for Endpoint {
    fn drop(&mut self) {
        if let Ok(m) = self.path.symlink_metadata()
            && (m.dev(), m.ino()) == self.identity
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
fn same_user(socket: &UnixStream) -> bool {
    unsafe {
        let mut cred: libc::ucred = std::mem::zeroed();
        let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        libc::getsockopt(
            socket.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut cred as *mut libc::ucred).cast(),
            &mut len,
        ) == 0
            && len as usize == std::mem::size_of::<libc::ucred>()
            && cred.uid == libc::geteuid()
    }
}

struct Packet {
    bytes: Vec<u8>,
    offset: usize,
    release: Option<String>,
}
struct Client {
    key: u64,
    _socket: Option<UnixStream>,
    input_fd: RawFd,
    output_fd: RawFd,
    input: Vec<u8>,
    partial_since: Option<Instant>,
    pending: VecDeque<(Frame, Instant)>,
    outstanding: HashSet<String>,
    output: VecDeque<Packet>,
    output_bytes: usize,
    write_since: Option<Instant>,
    eof: bool,
    dead: bool,
    framing_error: bool,
    blocked_search: bool,
}
impl Client {
    fn new(key: u64, socket: Option<UnixStream>) -> Self {
        let input_fd = socket
            .as_ref()
            .map_or(libc::STDIN_FILENO, AsRawFd::as_raw_fd);
        let output_fd = socket
            .as_ref()
            .map_or(libc::STDOUT_FILENO, AsRawFd::as_raw_fd);
        Self {
            key,
            _socket: socket,
            input_fd,
            output_fd,
            input: vec![],
            partial_since: None,
            pending: VecDeque::new(),
            outstanding: HashSet::new(),
            output: VecDeque::new(),
            output_bytes: 0,
            write_since: None,
            eof: false,
            dead: false,
            framing_error: false,
            blocked_search: false,
        }
    }
    fn enqueue(&mut self, bytes: Vec<u8>, release: Option<String>) {
        if self.output_bytes.saturating_add(bytes.len()) > MAX_RESPONSE {
            self.dead = true;
            return;
        }
        if self.output.is_empty() {
            self.write_since = Some(Instant::now());
        }
        self.output_bytes += bytes.len();
        self.output.push_back(Packet {
            bytes,
            offset: 0,
            release,
        });
    }
    fn fail(&mut self, id: Option<&str>, code: &str, message: &str, exit: u8) {
        self.enqueue(
            protocol::bytes(&protocol::failure(id, code, message, exit)).unwrap(),
            id.map(str::to_owned),
        );
    }
    fn frame_error(&mut self, message: &str) {
        self.framing_error = true;
        self.fail(None, "invalid_frame", message, 2);
        self.eof = true;
        self.input.clear();
        self.partial_since = None;
    }
    fn write(&mut self) {
        if let Some(packet) = self.output.front_mut()
            && ready(self.output_fd, libc::POLLOUT)
        {
            match write_fd(self.output_fd, &packet.bytes[packet.offset..]) {
                Ok(0) => self.dead = true,
                Ok(n) => {
                    packet.offset += n;
                    self.write_since = Some(Instant::now());
                    if packet.offset == packet.bytes.len() {
                        let packet = self.output.pop_front().unwrap();
                        // Credits track retained allocations, including prefixes
                        // already written. Release only when the packet is freed.
                        self.output_bytes -= packet.bytes.len();
                        if let Some(id) = packet.release {
                            self.outstanding.remove(&id);
                        }
                        if self.output.is_empty() {
                            self.write_since = None;
                        }
                    }
                }
                Err(e) if transient(&e) => {}
                Err(_) => self.dead = true,
            }
        }
        if self.write_since.is_some_and(|t| t.elapsed() > IO_TIMEOUT) {
            self.dead = true;
        }
    }
    fn read(&mut self) {
        if self._socket.is_some() && ready(self.input_fd, libc::POLLHUP) {
            self.dead = true;
        }
        if self.eof || self.dead || self.input.contains(&b'\n') {
            return;
        }
        if ready(self.input_fd, libc::POLLIN) {
            let mut buf = [0; 8192];
            let room = MAX_FRAME.saturating_sub(self.input.len()).min(buf.len());
            if room == 0 {
                self.frame_error("JSONL frame exceeds 128 KiB");
                return;
            }
            match read_fd(self.input_fd, &mut buf[..room]) {
                Ok(0) => {
                    self.eof = true;
                }
                Ok(n) => {
                    if self.input.is_empty() {
                        self.partial_since = Some(Instant::now());
                    }
                    self.input.extend_from_slice(&buf[..n]);
                }
                Err(e) if transient(&e) => {}
                Err(_) => self.dead = true,
            }
        }
        if !self.input.contains(&b'\n')
            && self.partial_since.is_some_and(|t| t.elapsed() > IO_TIMEOUT)
        {
            self.frame_error("partial frame timed out");
        }
    }
}

struct Backend {
    child: Child,
    socket: UnixStream,
    send: Vec<u8>,
    sent: usize,
    response: Vec<u8>,
    line_start: usize,
    created: Instant,
}
impl Backend {
    fn spawn(config: &Config) -> Result<Self> {
        let (socket, child_socket) = UnixStream::pair()?;
        let stdin = unsafe { File::from_raw_fd(child_socket.try_clone()?.into_raw_fd()) };
        let stdout = unsafe { File::from_raw_fd(child_socket.into_raw_fd()) };
        let parent = unsafe { libc::getpid() };
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg("__query-engine")
            .stdin(Stdio::from(stdin))
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::inherit());
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                }
                Ok(())
            });
        }
        let send = protocol::bytes(config)?;
        ensure!(
            send.len() <= 1024 * 1024,
            "engine configuration exceeds 1 MiB"
        );
        let child = command.spawn().context("start query engine child")?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            child,
            socket,
            send,
            sent: 0,
            response: vec![],
            line_start: 0,
            created: Instant::now(),
        })
    }
    fn submit(&mut self, frame: &Frame) -> Result<()> {
        self.send.extend(protocol::bytes(frame)?);
        self.response.clear();
        self.line_start = 0;
        Ok(())
    }
    /// Returns a complete bounded response and whether real dense encoding ran.
    fn poll(&mut self, id: &str) -> Result<Option<(Vec<u8>, bool, bool)>> {
        if self.sent < self.send.len() {
            match self.socket.write(&self.send[self.sent..]) {
                Ok(0) => anyhow::bail!("engine input closed"),
                Ok(n) => {
                    self.sent += n;
                    if self.sent == self.send.len() {
                        self.send.clear();
                        self.sent = 0;
                    }
                }
                Err(e) if transient(&e) => {}
                Err(e) => return Err(e.into()),
            }
        }
        let mut buf = [0; 16384];
        match self.socket.read(&mut buf) {
            Ok(0) => anyhow::bail!("query engine exited before completing the request"),
            Ok(n) => {
                ensure!(
                    n <= MAX_RESPONSE.saturating_sub(self.response.len()),
                    "response_limit: engine response exceeds 8 MiB"
                );
                self.response.extend_from_slice(&buf[..n]);
                while let Some(end) = self.response[self.line_start..]
                    .iter()
                    .position(|&b| b == b'\n')
                {
                    let end = self.line_start + end + 1;
                    let event: Value =
                        serde_json::from_slice(&self.response[self.line_start..end])?;
                    ensure!(
                        event["v"] == 1 && event["id"].as_str() == Some(id),
                        "invalid engine response attribution"
                    );
                    self.line_start = end;
                    if matches!(
                        event["type"].as_str(),
                        Some("search_completed" | "request_failed")
                    ) {
                        ensure!(
                            end == self.response.len(),
                            "unexpected data after engine terminal event"
                        );
                        return Ok(Some((
                            std::mem::take(&mut self.response),
                            event["query_provider"].is_string(),
                            event["type"] == "request_failed",
                        )));
                    }
                }
            }
            Err(e) if transient(&e) => {}
            Err(e) => return Err(e.into()),
        }
        Ok(None)
    }
}
impl Drop for Backend {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
struct Active {
    client: u64,
    id: String,
    started: Instant,
}

pub fn default_socket() -> Result<PathBuf> {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .context("set an absolute XDG_RUNTIME_DIR or pass --socket in a private directory")?;
    let m = runtime.metadata()?;
    ensure!(
        m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
        "unsafe XDG_RUNTIME_DIR"
    );
    let parent = runtime.join("ree");
    use std::os::unix::fs::DirBuilderExt;
    match std::fs::DirBuilder::new().mode(0o700).create(&parent) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    Ok(parent.join("worker.sock"))
}

pub fn serve(
    config: Config,
    path: Option<PathBuf>,
    idle: Duration,
    events: &mut Events,
) -> Result<u8> {
    ensure!(!idle.is_zero(), "idle timeout must be positive");
    let _signals = Signals::install()?;
    let endpoint = path.as_ref().map(|p| Endpoint::bind(p)).transpose()?;
    let stream = endpoint.is_none();
    let _stdout_nonblocking = if stream {
        Some(Nonblocking::new(libc::STDOUT_FILENO)?)
    } else {
        None
    };
    let mut clients = if stream {
        vec![Client::new(0, None)]
    } else {
        vec![]
    };
    if let Some(endpoint) = &endpoint {
        events.emit(json!({"type":"worker_started","v":1,"id":null,"socket":endpoint.path,"pid":std::process::id(),"idle_timeout_seconds":idle.as_secs(),"max_clients":MAX_CLIENTS,"engine":"disposable_process"}))?;
        events.flush()?;
    }
    let mut next_key = 1;
    let mut backend: Option<Backend> = None;
    let mut active: Option<Active> = None;
    let mut last_dense: Option<Instant> = None;
    let mut last_client = 0;
    let mut exit = 0;
    let mut stopping = None;
    loop {
        if STOP.load(Ordering::Relaxed) && stopping.is_none() {
            if active.is_some() || clients.iter().any(|c| !c.pending.is_empty()) {
                exit = exit.max(1);
            }
            stopping = Some(Instant::now());
            backend.take();
            if let Some(a) = active.take()
                && let Some(c) = clients.iter_mut().find(|c| c.key == a.client)
            {
                c.fail(Some(&a.id), "cancelled", "worker is shutting down", 3);
            }
            for c in &mut clients {
                while let Some((f, _)) = c.pending.pop_front() {
                    c.fail(Some(&f.id), "cancelled", "worker is shutting down", 3);
                }
                c.eof = true;
                c.input.clear();
            }
        }
        if stopping.is_none()
            && let Some(endpoint) = &endpoint
        {
            // Bound accept work so a connection flood cannot starve inference.
            for _ in 0..MAX_CLIENTS {
                match endpoint.listener.accept() {
                    Ok((mut socket, _)) => {
                        socket.set_nonblocking(true)?;
                        if clients.len() >= MAX_CLIENTS || !same_user(&socket) {
                            let _ = socket.write(&protocol::bytes(&protocol::failure(
                                None,
                                "busy",
                                "worker client limit or peer rejection",
                                3,
                            ))?);
                        } else {
                            clients.push(Client::new(next_key, Some(socket)));
                            next_key += 1;
                        }
                    }
                    Err(e) if transient(&e) => break,
                    Err(e) => return Err(e.into()),
                }
            }
        }
        for c in &mut clients {
            c.write();
            // Peek one bounded frame beyond the search queue so a following
            // cancellation is not blocked by two pending searches. An extra
            // search stays in the input buffer and backpressures stdin; do not
            // repeatedly parse/allocate it while waiting for capacity.
            if c.pending.len() < QUEUE_PER_CLIENT {
                c.blocked_search = false;
            }
            if !c.blocked_search {
                c.read();
            }
            if !c.dead && stopping.is_none() {
                for _ in 0..4 {
                    if c.blocked_search {
                        break;
                    }
                    let Some(end) = c.input.iter().position(|&b| b == b'\n') else {
                        break;
                    };
                    let frame = match serde_json::from_slice::<Frame>(&c.input[..=end]) {
                        Ok(f) => f,
                        Err(e) => {
                            c.frame_error(&format!("invalid JSONL request: {e}"));
                            exit = exit.max(2);
                            break;
                        }
                    };
                    if stream
                        && c.pending.len() >= QUEUE_PER_CLIENT
                        && matches!(frame.operation, Operation::Search { .. })
                    {
                        c.blocked_search = true;
                        break;
                    }
                    c.input.drain(..=end);
                    c.partial_since = (!c.input.is_empty()).then(Instant::now);
                    if !protocol::valid_id(&frame.id) {
                        c.frame_error("invalid request id");
                        exit = exit.max(2);
                        break;
                    }
                    if c.outstanding.contains(&frame.id) {
                        // A duplicate id cannot receive a second terminal result.
                        c.frame_error("duplicate outstanding request id");
                        exit = exit.max(2);
                        break;
                    }
                    c.outstanding.insert(frame.id.clone());
                    if let Err(e) = frame.validate() {
                        c.fail(Some(&frame.id), "invalid_request", &format!("{e:#}"), 2);
                        exit = exit.max(1);
                        continue;
                    }
                    match &frame.operation {
                        Operation::Cancel { target } => {
                            let queued = c.pending.iter().position(|(f, _)| &f.id == target);
                            let running = active
                                .as_ref()
                                .is_some_and(|a| a.client == c.key && &a.id == target);
                            if let Some(index) = queued {
                                c.pending.remove(index);
                                c.fail(
                                    Some(target),
                                    "cancelled",
                                    "request cancelled before execution",
                                    3,
                                );
                                exit = exit.max(1);
                            } else if running {
                                backend.take();
                                active.take();
                                last_dense = None;
                                c.fail(
                                    Some(target),
                                    "cancelled",
                                    "engine process terminated; next query reloads",
                                    3,
                                );
                                exit = exit.max(1);
                            }
                            c.enqueue(protocol::bytes(&json!({"v":1,"id":frame.id,"type":"cancel_completed","target":target,"status":if queued.is_some()||running{"cancelled"}else{"not_pending"}}))?,Some(frame.id.clone()));
                        }
                        Operation::Search { .. } => {
                            if c.pending.len() >= QUEUE_PER_CLIENT {
                                c.fail(Some(&frame.id), "busy", "worker request queue is full", 3);
                                exit = exit.max(1);
                            } else {
                                c.pending.push_back((frame, Instant::now()));
                            }
                        }
                    }
                }
            }
            if c.eof && !c.input.is_empty() && !c.input.contains(&b'\n') {
                c.frame_error("partial final JSONL frame");
                exit = exit.max(2);
            }
        }
        if clients.iter().any(|c| c.framing_error) {
            exit = exit.max(2);
        }
        if stream && clients.iter().any(|c| c.dead) {
            exit = exit.max(3);
        }
        // Bound aggregate slow-consumer/control output as well as each client.
        while clients
            .iter()
            .filter(|c| !c.dead)
            .map(|c| c.output_bytes)
            .sum::<usize>()
            > MAX_RESPONSE
        {
            if let Some(c) = clients
                .iter_mut()
                .filter(|c| !c.dead)
                .max_by_key(|c| c.output_bytes)
            {
                c.dead = true;
            } else {
                break;
            }
        }
        // Cancel a disconnected consumer instead of retaining its GPU work.
        if active
            .as_ref()
            .is_some_and(|a| clients.iter().any(|c| c.key == a.client && c.dead))
        {
            active.take();
            backend.take();
            last_dense = None;
        }
        if let (Some(a), Some(engine)) = (&active, &mut backend) {
            match engine.poll(&a.id) {
                Ok(Some((response, dense, failed))) => {
                    if dense {
                        last_dense = Some(Instant::now());
                    }
                    if let Some(c) = clients.iter_mut().find(|c| c.key == a.client) {
                        if failed {
                            exit = exit.max(1);
                        }
                        c.enqueue(response, Some(a.id.clone()));
                    }
                    active.take();
                }
                Err(e) => {
                    if let Some(c) = clients.iter_mut().find(|c| c.key == a.client) {
                        c.fail(Some(&a.id), "engine_failed", &format!("{e:#}"), 3);
                    }
                    active.take();
                    backend.take();
                    last_dense = None;
                    exit = exit.max(1);
                }
                _ => {}
            }
        }
        // A malicious/stuck request cannot retain a child indefinitely. This is
        // a process deadline, not a claim that an ONNX call is interruptible.
        if active
            .as_ref()
            .is_some_and(|a| a.started.elapsed() > Duration::from_secs(1800))
        {
            let a = active.take().unwrap();
            backend.take();
            last_dense = None;
            if let Some(c) = clients.iter_mut().find(|c| c.key == a.client) {
                c.fail(
                    Some(&a.id),
                    "deadline_exceeded",
                    "query exceeded 30-minute execution limit",
                    3,
                );
            }
            exit = exit.max(1);
        }
        let queued_dense = clients
            .iter()
            .any(|c| c.pending.iter().any(|(f, _)| f.dense()));
        if active.is_none()
            && !queued_dense
            && let Some(engine) = &backend
            && last_dense.unwrap_or(engine.created).elapsed() >= idle
        {
            backend.take();
            last_dense = None;
        }
        clients.retain(|c| {
            let drained = c.eof
                && c.input.is_empty()
                && c.pending.is_empty()
                && c.output.is_empty()
                && active.as_ref().is_none_or(|a| a.client != c.key);
            !(c.dead || drained)
        });
        if (stream || stopping.is_some()) && clients.is_empty() {
            break;
        }
        if stopping.is_some_and(|t| t.elapsed() > IO_TIMEOUT) {
            break;
        }
        // Reserve one response budget for the active child. At most 16 MiB of
        // pending serialized output exists across child and all consumers.
        if active.is_none()
            && stopping.is_none()
            && clients.iter().map(|c| c.output_bytes).sum::<usize>() <= MAX_RESPONSE
        {
            let index = clients
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.pending.is_empty() && c.output_bytes < MAX_RESPONSE / 2)
                .min_by_key(|(_, c)| (c.key <= last_client, c.key))
                .map(|(i, _)| i);
            if let Some(i) = index {
                let c = &mut clients[i];
                let (frame, queued) = c.pending.pop_front().unwrap();
                last_client = c.key;
                let result = (|| -> Result<()> {
                    if backend.is_none() {
                        backend = Some(Backend::spawn(&config)?);
                        last_dense = None;
                    }
                    backend.as_mut().unwrap().submit(&frame)
                })();
                match result {
                    Ok(()) => {
                        if crate::metrics::enabled() {
                            c.enqueue(protocol::bytes(&json!({"v":1,"id":frame.id,"type":"queue_timing","queue_wait_ns":queued.elapsed().as_nanos()}))?,None);
                        }
                        active = Some(Active {
                            client: c.key,
                            id: frame.id,
                            started: Instant::now(),
                        });
                    }
                    Err(e) => {
                        backend.take();
                        c.fail(Some(&frame.id), "engine_failed", &format!("{e:#}"), 3);
                        exit = exit.max(1);
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    drop(backend);
    Ok(if stream { exit } else { 0 })
}

pub fn client(path: &Path, request: super::SearchRequest, events: &mut Events) -> Result<u8> {
    request.validate()?;
    let mut socket = UnixStream::connect(path).with_context(|| {
        format!(
            "connect worker {}; start ree worker explicitly",
            path.display()
        )
    })?;
    ensure!(same_user(&socket), "worker peer UID differs from this user");
    socket.set_read_timeout(Some(Duration::from_secs(1800)))?;
    socket.set_write_timeout(Some(IO_TIMEOUT))?;
    let frame = Frame::search(uuid::Uuid::new_v4().to_string(), request);
    frame
        .validate()
        .map_err(|e| AppError::new(2, format!("{e:#}")))?;
    let encoded = protocol::bytes(&frame)?;
    if encoded.len() > MAX_FRAME {
        return Err(AppError::new(2, "request frame exceeds 128 KiB").into());
    }
    socket.write_all(&encoded)?;
    let mut reader = std::io::BufReader::new(socket);
    let mut remaining = MAX_RESPONSE;
    loop {
        let line = protocol::read_frame(&mut reader, remaining)?
            .context("worker disconnected before terminal response")?;
        remaining = remaining.saturating_sub(line.len());
        let mut event: Value = serde_json::from_slice(&line)?;
        if event["type"] == "connection_failed" {
            return Err(AppError::new(
                3,
                event["message"]
                    .as_str()
                    .unwrap_or("worker rejected connection"),
            )
            .into());
        }
        ensure!(
            event["v"] == 1 && event["id"].as_str() == Some(&frame.id),
            "worker response has wrong protocol/id"
        );
        let kind = event["type"].as_str().unwrap_or("").to_owned();
        if kind == "request_failed" {
            return Err(AppError::new(
                event["exit_code"].as_u64().unwrap_or(3) as u8,
                event["message"].as_str().unwrap_or("worker request failed"),
            )
            .into());
        }
        // Explicit CLI routing preserves ordinary search events/quiet semantics.
        event
            .as_object_mut()
            .context("worker event must be an object")?
            .remove("id");
        event.as_object_mut().unwrap().remove("v");
        events.emit(event)?;
        if kind == "search_completed" {
            return Ok(0);
        }
    }
}
