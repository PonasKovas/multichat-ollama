use std::io;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use tokio::fs;
use tokio_rustls::rustls::{self, ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Rustls(#[from] rustls::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub async fn configure(certificate: &Path) -> Result<TlsConnector, Error> {
    let certificates = fs::read(certificate).await?;
    let certificates = rustls_pemfile::certs(&mut &*certificates).collect::<Result<Vec<_>, _>>()?;

    let mut store = RootCertStore::empty();
    for certificate in certificates {
        store.add(certificate)?;
    }

    let config = ClientConfig::builder()
        .with_root_certificates(store)
        .with_no_client_auth();

    let config = Arc::new(config);

    Ok(TlsConnector::from(config))
}
