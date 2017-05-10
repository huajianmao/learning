use ifocus;

CREATE TABLE user (
  username VARCHAR(36) NOT NULL PRIMARY KEY,
  password VARCHAR(128) NOT NULL
);

CREATE TABLE token (
  id       INT PRIMARY KEY,
  username VARCHAR(36) NOT NULL,
  token    VARCHAR(128) NOT NULL,
  FOREIGN  KEY (username) REFERENCES user(username)
);
