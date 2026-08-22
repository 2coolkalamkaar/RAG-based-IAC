# Policy: No Public Access

## Scope
Applies to all S3 buckets, RDS instances, EC2 instances, and load balancers.

## S3 Buckets

All S3 buckets MUST have public access blocked:

```hcl
resource "aws_s3_bucket_public_access_block" "this" {
  bucket = aws_s3_bucket.this.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

`acl = "public-read"` or `acl = "public-read-write"` is NEVER permitted.

## RDS Instances

```hcl
resource "aws_db_instance" "this" {
  publicly_accessible = false   # required
  ...
}
```

## EC2 Instances

EC2 instances MUST NOT have `associate_public_ip_address = true` unless the user
explicitly requests a bastion host for SSH access. Even then, restrict SSH ingress
to the VPC CIDR only — never `0.0.0.0/0`.

## Security Groups — Egress Only

Security groups should allow ONLY the minimum required ingress traffic.
NEVER generate ingress rules with `cidr_blocks = ["0.0.0.0/0"]` on ports other
than 80 and 443 (HTTP/HTTPS for public-facing load balancers).

## Enforcement

Violations trigger a TFLint or Checkov finding that will fail validation and
route the code to the Fixer Node for remediation.
