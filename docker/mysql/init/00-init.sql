CREATE DATABASE cryptoml;
CREATE USER 'cryptoml-api'@'%' IDENTIFIED BY 'secret';
GRANT ALL PRIVILEGES ON `cryptoml`.* TO 'cryptoml-api'@'%';
GRANT USAGE ON `cryptoml`.* TO 'cryptoml-api'@'%' REQUIRE NONE WITH MAX_QUERIES_PER_HOUR 0 MAX_CONNECTIONS_PER_HOUR 0 MAX_UPDATES_PER_HOUR 0 MAX_USER_CONNECTIONS 0;