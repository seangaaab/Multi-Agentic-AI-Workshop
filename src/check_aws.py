import boto3


def main():
    sts = boto3.client("sts")
    ident = sts.get_caller_identity()
    print("AWS Account:", ident["Account"])
    print("Caller ARN:", ident["Arn"])


if __name__ == "__main__":
    main()
