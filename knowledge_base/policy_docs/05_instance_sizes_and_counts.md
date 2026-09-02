# Policy: Allowed Instance Sizes and Resource Counts

## Scope
EC2 instances, RDS instances, ElastiCache nodes, and any resource with a `count` or `for_each`.

## Allowed EC2 Instance Families

For general workloads:
- `t3.micro`, `t3.small`, `t3.medium`, `t3.large` — development/staging
- `m5.large`, `m5.xlarge`, `m5.2xlarge` — production workloads
- `c5.large`, `c5.xlarge` — compute-optimised
- `r5.large`, `r5.xlarge` — memory-optimised

**Restricted (require explicit justification):**
- Any instance type larger than `*.4xlarge`
- GPU instances (`p3`, `p4`, `g4`, `g5`)
- Metal instances

## Allowed RDS Instance Classes

- `db.t3.micro` through `db.t3.medium` — development
- `db.m5.large`, `db.m5.xlarge` — production

Multi-AZ is recommended for production RDS. Single-AZ is acceptable for dev/demo.

## Resource Count Safety Limits

The agent MUST NOT generate resources with `count > 20` unless the user's request
explicitly states a larger number AND the request is clearly intentional (e.g.,
"50 spot instances for batch processing").

If a hallucinated count exceeds 20, the Architect MUST reduce it to the minimum
implied by the request (typically 1-3) and note the reduction in the Change Summary.

This is a guard against runaway cost from `count = 500` typos or hallucinations
generating 500 real EC2 instances on apply.

## Allowed AWS Regions

For this agent deployment: `us-east-1`, `us-west-2`, `eu-west-1`.
Do NOT generate provider blocks with regions outside this list without explicit user request.

```hcl
provider "aws" {
  region = "us-east-1"  # default if user doesn't specify
}
```
