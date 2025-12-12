CREATE DATABASE pet_ai;
USE pet_ai;
CREATE TABLE bcs_history (
    bcs_id INT AUTO_INCREMENT PRIMARY KEY,
    pet_id INT NOT NULL,
    bcs_value INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
