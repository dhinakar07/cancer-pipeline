# AWS Setup Guide — cancer-pipeline

Step-by-step instructions to provision the AWS infrastructure for this project.
Everything is Free Tier eligible for small workloads.

---

## What We're Creating

| AWS Service | Purpose | Free Tier? |
|---|---|---|
| S3 Bucket | Store raw data, processed snapshots, model artefacts | ✅ 5 GB free |
| RDS PostgreSQL | Cloud-managed database (replaces local Docker) | ✅ 750 hrs/month (db.t3.micro) |
| IAM User + Policy | Secure credentials for local dev | ✅ Always free |
| IAM Role | Credential-free access for EC2/MWAA | ✅ Always free |

---

## Step 1 — Install & Configure AWS CLI

```bash
# Install
pip install awscli

# Configure with your IAM credentials
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region:        us-east-1
# Default output format: json

# Verify
aws sts get-caller-identity
```

---

## Step 2 — Create an S3 Bucket

```bash
# Replace 'your-cancer-pipeline' with a globally unique name
# (no underscores, all lowercase)
BUCKET_NAME="your-cancer-pipeline-$(whoami)"

aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Enable versioning so you never lose a model or dataset
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled

# Verify
aws s3 ls s3://$BUCKET_NAME
```

Then add to your `.env`:
```
S3_BUCKET=your-cancer-pipeline-yourname
AWS_REGION=us-east-1
```

---

## Step 3 — Upload Your Model Artefacts to S3

```bash
# From the project root — do this once after training
python -c "
from src.cloud.s3_handler import upload_model_artefacts
result = upload_model_artefacts()
print(result)
"
# Expected: {'model_uploaded': True, 'tokenizer_uploaded': True}
```

Now any machine with S3 access can run the pipeline without copying model files manually.

---

## Step 4 — Create RDS PostgreSQL (optional)

### 4a. Create a Security Group

```bash
# Create a security group for the RDS instance
SG_ID=$(aws ec2 create-security-group \
    --group-name cancer-pipeline-rds-sg \
    --description "RDS access for cancer-pipeline" \
    --query 'GroupId' --output text)

# Allow inbound PostgreSQL (port 5432) from your IP only
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 5432 \
    --cidr "$MY_IP/32"

echo "Security Group ID: $SG_ID"
```

### 4b. Create the RDS Instance

```bash
aws rds create-db-instance \
    --db-instance-identifier cancer-pipeline-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 16.2 \
    --master-username cancer_user \
    --master-user-password "YourSecurePassword123!" \
    --db-name cancer_pipeline \
    --allocated-storage 20 \
    --storage-type gp2 \
    --vpc-security-group-ids $SG_ID \
    --publicly-accessible \
    --no-multi-az \
    --region us-east-1
```

This takes ~5 minutes. Wait until status is `available`:
```bash
aws rds wait db-instance-available \
    --db-instance-identifier cancer-pipeline-db

# Get the endpoint
aws rds describe-db-instances \
    --db-instance-identifier cancer-pipeline-db \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text
```

### 4c. Update .env with the RDS endpoint

```
RDS_HOST=cancer-pipeline-db.xxxx.us-east-1.rds.amazonaws.com
RDS_PORT=5432
POSTGRES_USER=cancer_user
POSTGRES_PASSWORD=YourSecurePassword123!
POSTGRES_DB=cancer_pipeline
```

### 4d. Initialise the schema on RDS

```bash
# Run the schema SQL against RDS (same file used for local Docker)
psql -h $RDS_HOST -U cancer_user -d cancer_pipeline \
    -f src/db/schema.sql
```

---

## Step 5 — Create an IAM Policy & User (for local dev)

```bash
# Create a least-privilege policy for S3 and RDS access
aws iam create-policy \
    --policy-name cancer-pipeline-policy \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:DeleteObject"
                ],
                "Resource": [
                    "arn:aws:s3:::your-cancer-pipeline-bucket",
                    "arn:aws:s3:::your-cancer-pipeline-bucket/*"
                ]
            }
        ]
    }'

# Create a dedicated IAM user for local development
aws iam create-user --user-name cancer-pipeline-dev

# Attach the policy
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws iam attach-user-policy \
    --user-name cancer-pipeline-dev \
    --policy-arn "arn:aws:iam::$ACCOUNT_ID:policy/cancer-pipeline-policy"

# Create access keys (save these in .env — never commit to git!)
aws iam create-access-key --user-name cancer-pipeline-dev
```

---

## Step 6 — Create an IAM Role (for EC2 / MWAA production)

In production, **never use access keys**. Attach an IAM Role instead:

```bash
# Create the role with EC2 trust policy
aws iam create-role \
    --role-name cancer-pipeline-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach the S3 policy
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws iam attach-role-policy \
    --role-name cancer-pipeline-role \
    --policy-arn "arn:aws:iam::$ACCOUNT_ID:policy/cancer-pipeline-policy"

# Create an instance profile and attach the role
aws iam create-instance-profile \
    --instance-profile-name cancer-pipeline-profile
aws iam add-role-to-instance-profile \
    --instance-profile-name cancer-pipeline-profile \
    --role-name cancer-pipeline-role
```

When this role is attached to your EC2 instance, the pipeline uses it
automatically — **no AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY needed.**

---

## Step 7 — Verify Cloud Mode End-to-End

```bash
# Run the pipeline in cloud mode (S3 + RDS)
python -c "
from src.config import cloud_enabled, POSTGRES_HOST
print('Cloud mode:', cloud_enabled())
print('DB host:', POSTGRES_HOST)

from src.db.connection import test_connection
test_connection()
"

# Then trigger the Airflow DAG and watch the logs
airflow dags trigger cancer_pipeline
airflow tasks logs cancer_pipeline upload_raw_to_s3 <run_id>
```

---

## Cost Estimate (Free Tier)

| Resource | Free Tier Limit | Estimated Usage |
|---|---|---|
| S3 Storage | 5 GB | ~50 MB (dataset + model) |
| S3 Requests | 20,000 GET / 2,000 PUT | ~50 per run |
| RDS db.t3.micro | 750 hours/month | 24 hrs/day = 720 hrs/month |
| RDS Storage | 20 GB | ~1 GB |
| **Estimated monthly cost** | | **$0 (within Free Tier)** |

**Note:** RDS Free Tier is only available for the first 12 months of a new AWS account.

---

## Cleanup (avoid charges)

```bash
# Delete RDS instance
aws rds delete-db-instance \
    --db-instance-identifier cancer-pipeline-db \
    --skip-final-snapshot

# Empty and delete S3 bucket
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME
```
