mod config;
mod handle_mc_update;
mod handle_ollama_gen;
mod ollama_api;
mod room_state;
mod tls;

use anyhow::Context;
use clap::Parser;
use config::Config;
use futures::future::FutureExt;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use handle_mc_update::handle_mc_update;
use handle_ollama_gen::handle_ollama_gen;
use multichat_client::proto::Config as ProtoConfig;
use multichat_client::{ClientBuilder, EitherStream, Update};
use room_state::{Message, RoomState};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio::{fs, select};
use tokio_rustls::client::TlsStream;
use tracing::{error, info, subscriber};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::{fmt, prelude::*};

#[derive(Parser)]
struct Args {
    #[clap(help = "Path to config file")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{}", e);
            ExitCode::FAILURE
        }
    }
}

async fn run() -> anyhow::Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().without_time().with_target(false));
    subscriber::set_global_default(registry).unwrap();

    let args = Args::parse();

    info!("Reading config from {}", args.config.display());

    let config = fs::read_to_string(&args.config)
        .await
        .context("reading config")?;
    let config = toml::from_str::<Config>(&config).context("parsing config")?;

    let mut state = State::create(config).await.context("initialization")?;

    info!("Connected to Multichat");

    loop {
        // we either wait for an update from multichat or
        // the Ollama endpoint to finish generating a response in any of the groups
        enum EventType {
            FinishGenerate {
                gid: u32,
                res: anyhow::Result<String>,
            },
            Multichat {
                update: Update,
            },
        }

        let event = {
            let mut ollama_api_tasks: FuturesUnordered<_> = state
                .rooms
                .iter_mut()
                .filter_map(|(gid, room)| {
                    room.ollama_api_task
                        .as_mut()
                        .map(|join| join.map(|r| (*gid, r)))
                })
                .collect();

            select! {
                Some((gid, res)) = ollama_api_tasks.next(), if !ollama_api_tasks.is_empty() => {
                    let res = res.unwrap(); // we unwrap the JoinError, since it would only be err if it panicked
                    EventType::FinishGenerate { gid, res }
                }
                update = state.mc_client.read_update() => {
                    EventType::Multichat { update: update.context("multichat update")? }
                }
            }
        };

        match event {
            EventType::Multichat { update } => {
                handle_mc_update(&mut state, update).await?;
            }
            EventType::FinishGenerate { gid, res } => {
                handle_ollama_gen(&mut state, gid, res).await?;
            }
        }
    }
}

struct State {
    mc_client: multichat_client::Client<EitherStream<TlsStream<TcpStream>>>,
    reqw: reqwest::Client,
    config: Config,

    // group id -> room data
    rooms: HashMap<u32, RoomState>,
}

impl State {
    pub async fn create(config: Config) -> anyhow::Result<Self> {
        let mut memories: HashMap<String, Vec<String>> = serde_json::from_str(
            &fs::read_to_string(&config.ollama.memory_file)
                .await
                .context("reading memory file")?,
        )
        .context("parsing memory file")?;

        let mc_connector = match &config.multichat.certificate {
            Some(certificate) => Some(tls::configure(certificate).await.context("TLS init")?),
            None => None,
        };

        let mut proto_config = ProtoConfig::default();
        proto_config.max_size(512 * 1024 * 1024); // 512 MiB

        let (groups, mut mc_client) = timeout(
            Duration::from_secs(5),
            ClientBuilder::maybe_tls(mc_connector)
                .config(proto_config)
                .connect(&config.multichat.server, config.multichat.access_token),
        )
        .await
        .context("connection timed out")?
        .context("connection to multichat")?;

        let mut rooms = HashMap::new();
        for group_name in &config.multichat.groups {
            let gid = *groups.get(group_name.as_str()).context("Group not found")?;
            mc_client.join_group(gid).await?;

            let my_uid = mc_client
                .join_user(gid, &config.multichat.user_name)
                .await?;

            rooms.insert(
                gid,
                RoomState::new(
                    my_uid,
                    group_name.clone(),
                    memories.remove(group_name).unwrap_or(Vec::new()),
                ),
            );
        }

        Ok(Self {
            mc_client,
            reqw: reqwest::Client::new(),
            config,
            rooms,
        })
    }
    pub async fn add_memory(&mut self, gid: u32, memory: String) -> anyhow::Result<()> {
        self.rooms.get_mut(&gid).unwrap().memories.push(memory);

        // save
        self.save_memories().await
    }
    pub async fn remove_memory(&mut self, gid: u32, idx: usize) -> anyhow::Result<String> {
        let mem = self.rooms.get_mut(&gid).unwrap().memories.remove(idx);

        // save
        self.save_memories().await?;

        Ok(mem)
    }
    async fn save_memories(&self) -> anyhow::Result<()> {
        let all_memories: HashMap<String, Vec<String>> = self
            .rooms
            .iter()
            .map(|(_gid, room)| (room.room_name.clone(), room.memories.clone()))
            .collect();
        fs::write(
            &self.config.ollama.memory_file,
            &serde_json::to_string_pretty(&all_memories)?,
        )
        .await?;

        Ok(())
    }
    pub fn push_message(&mut self, gid: u32, msg: Message) {
        let room = self.rooms.get_mut(&gid).unwrap();

        if room.message_history.len() == self.config.ollama.prompt_messages_n {
            room.message_history.pop_front();
        }
        room.message_history.push_back(msg);
    }
}
