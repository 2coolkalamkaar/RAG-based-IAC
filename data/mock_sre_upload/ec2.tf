# ============================================================
# ec2.tf — Application EC2 Instance (Web Server)
# Known Issues (intentional for HitL demo):
#   - Hardcoded AMI ID (not data source lookup)
#   - No IAM instance profile / role attached
#   - Root EBS volume is unencrypted
#   - No ebs_optimized flag
# ============================================================

resource "aws_instance" "web_server" {
  ami           = "ami-0c02fb55956c7d316"  # Amazon Linux 2 us-east-1 (hardcoded)
  instance_type = "t2.micro"               # SRE: consider upgrading to t3.medium for prod

  subnet_id              = aws_subnet.public_a.id
  vpc_security_group_ids = [aws_security_group.web_sg.id]

  # WARNING: No IAM instance profile attached
  # iam_instance_profile = ""  # TODO: create and attach

  root_block_device {
    volume_type = "gp2"
    volume_size = 20
    # encrypted = true  # TODO: Enable encryption
  }

  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install -y httpd
    systemctl start httpd
    systemctl enable httpd
    echo "<h1>Production Web Server</h1>" > /var/www/html/index.html
  EOF

  tags = {
    Name        = "prod-web-server"
    Environment = "production"
    Role        = "web"
  }
}

# Elastic IP for the web server
resource "aws_eip" "web_server" {
  instance = aws_instance.web_server.id
  domain   = "vpc"

  tags = {
    Name = "prod-web-eip"
  }
}
