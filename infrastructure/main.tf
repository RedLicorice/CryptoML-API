 provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "docker-desktop"
}

resource "kubernetes_namespace" "cryptoml" {
  metadata {
    name = "cryptoml"
  }
}