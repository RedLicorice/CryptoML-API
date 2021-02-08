CREATE DATABASE cryptoml;
CREATE USER 'cryptoml-api'@'%' IDENTIFIED BY 'secret';
GRANT USAGE ON *.* TO 'cryptoml-api'@'%' REQUIRE NONE WITH MAX_QUERIES_PER_HOUR 0 MAX_CONNECTIONS_PER_HOUR 0 MAX_UPDATES_PER_HOUR 0 MAX_USER_CONNECTIONS 0;
GRANT ALL PRIVILEGES ON `cryptoml`.* TO 'cryptoml-api'@'%';