"""
Lambda function: ODM stage-switch orchestrator.

Triggered by EC2 instance state-change (stopped) via EventBridge.
Reads the next instance type from an instance tag 'odm:next-instance-type',
modifies the instance type, clears the tag, and starts the instance.

If the tag is absent or empty, does nothing (normal shutdown, not a stage switch).

Environment variables (set by Terraform):
    INSTANCE_ID: EC2 instance ID to manage
"""

import json
import os
import time

import boto3

ec2 = boto3.client("ec2")


def handler(event, context):
    instance_id = os.environ.get("INSTANCE_ID")
    if not instance_id:
        print("No INSTANCE_ID env var — skipping")
        return

    # Verify instance is stopped
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
    if state != "stopped":
        print(f"Instance {instance_id} is {state}, not stopped — skipping")
        return

    # Read the next-instance-type tag
    tags = resp["Reservations"][0]["Instances"][0].get("Tags", [])
    next_type_tag = next((t["Value"] for t in tags if t["Key"] == "odm:next-instance-type"), "")
    if not next_type_tag:
        print(f"No odm:next-instance-type tag on {instance_id} — normal shutdown, skipping")
        return

    current_type = resp["Reservations"][0]["Instances"][0]["InstanceType"]
    print(f"Stage switch: {current_type} → {next_type_tag}")

    # Modify instance type
    ec2.modify_instance_attribute(
        InstanceId=instance_id,
        InstanceType={"Value": next_type_tag},
    )
    print(f"Modified instance type to {next_type_tag}")

    # Clear the tag so the next shutdown doesn't re-trigger
    ec2.delete_tags(
        Resources=[instance_id],
        Tags=[{"Key": "odm:next-instance-type"}],
    )

    # Start the instance
    ec2.start_instances(InstanceIds=[instance_id])
    print(f"Started instance {instance_id} as {next_type_tag}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "action": "stage-switch",
            "from": current_type,
            "to": next_type_tag,
            "instance": instance_id,
        }),
    }
