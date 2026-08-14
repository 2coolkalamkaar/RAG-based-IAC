# ============================================================
# security_groups.tf — Web & DB Security Groups
# Known Issues (intentional for HitL demo):
#   - SSH (port 22) open to the world (0.0.0.0/0)
#   - No egress restrictions (all traffic allowed out)
#   - DB security group allows 3306 from entire VPC CIDR
# ============================================================

# --- Web Server Security Group ---
resource "aws_security_group" "web_sg" {
  name        = "prod-web-sg"
  description = "Security group for production web server"
  vpc_id      = aws_vpc.main.id

  # HTTP - Public
  ingress {
    description = "HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS - Public
  ingress {
    description = "HTTPS from anywhere"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # SSH - OPEN TO THE WORLD (security risk - good for HitL demo)
  ingress {
    description = "SSH - TEMPORARY - restrict this!"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # TODO: Lock down to bastion host IP
  }

  # Egress - All traffic allowed (unrestricted)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "prod-web-sg"
    Environment = "production"
  }
}

# --- Database Security Group ---
resource "aws_security_group" "db_sg" {
  name        = "prod-db-sg"
  description = "Security group for production RDS MySQL"
  vpc_id      = aws_vpc.main.id

  # MySQL - from entire VPC CIDR (too broad)
  ingress {
    description = "MySQL from VPC"
    from_port   = 3306
    to_port     = 3306
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # TODO: restrict to web_sg ID only
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "prod-db-sg"
    Environment = "production"
  }
}
