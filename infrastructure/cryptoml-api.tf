resource "kubernetes_deployment" "cryptoml-api" {
    metadata {
        name = "cryptoml-api"
        labels = {
            app   = "api"
        }
    }

    spec {
        selector {
            match_labels = {
                app   = "api"
            }
        }
        #Number of replicas
        replicas = 1
        #Template for the creation of the pod
        template {
            metadata {
                labels = {
                    app   = "api"
                }
            }
            spec {
                container {
                    image = "mrlicorice/cryptoml-api"
                    name  = "cryptoml-api"
                    command = ["python3", "app.py"]

                    #List of ports to expose from the container.
                    port {
                        container_port = 8000
                    }
                }
            }
        }
    }
}