use multichat_client::proto::AccessToken;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::PathBuf;
use url::Url;

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    pub multichat: Multichat,
    pub ollama: Ollama,
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Multichat {
    pub server: String,
    pub access_token: AccessToken,
    pub certificate: Option<PathBuf>,
    pub user_name: String,
    pub groups: HashSet<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Ollama {
    pub memory_file: PathBuf,
    pub basic_auth_user: String,
    pub basic_auth_password: String,
    pub base_url: Url,
    pub mention_name: String,
    pub model: String,
    pub system_prompt: String,
    pub prompt_messages_n: usize,
    pub temperature: f32,
    pub top_k: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_parses() {
        let config = include_str!("../example/config.toml");
        toml::from_str::<Config>(config).unwrap();
    }
}
