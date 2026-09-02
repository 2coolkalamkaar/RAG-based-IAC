# Policy: Encryption at Rest — Mandatory

## Scope
All storage resources: S3, EBS, RDS, DynamoDB, EFS, Secrets Manager.

## S3 Buckets

```hcl
resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  bucket = aws_s3_bucket.this.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

## EBS Volumes

All EBS root volumes and additional volumes MUST have encryption enabled:

```hcl
resource "aws_ebs_volume" "this" {
  encrypted = true
  ...
}
```

In launch templates and instance configs:
```hcl
block_device_mappings {
  device_name = "/dev/xvda"
  ebs {
    encrypted             = true
    volume_type           = "gp3"
    delete_on_termination = true
  }
}
```

## RDS

```hcl
resource "aws_db_instance" "this" {
  storage_encrypted = true
  ...
}
```

## DynamoDB

DynamoDB encrypts at rest by default (AWS-owned keys). No additional config needed unless
the user requests KMS CMK encryption:
```hcl
server_side_encryption {
  enabled = true
}
```

## Enforcement

Checkov rules `CKV_AWS_17`, `CKV_AWS_19`, `CKV_AWS_74`, `CKV_AWS_86` cover these requirements
and will fail the Secure RAG validation step if violated.
