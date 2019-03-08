import okera
import sys

# pylint: disable=line-too-long
NIGHTLY_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzUxMiJ9.eyJzdWIiOiJjZXJlYnJvIiwiaXNzIjoiY2VyZWJyb2RhdGEuY29tIiwiZ3JvdXBzIjpbInJvb3QiLCJhZG1pbiJdLCJleHAiOjE1OTA1MTA4MDd9.diIN9SfRgaIiMEx9tEBM4vksuxCAE1l9fLH-J7qtJxUu0oLrAXFW6BiJrFt6-YdlDmAQhbB5Q7zGiBv8uKrq8tvaqpzV-16IRDgZ3SQVh4SklE5G5SX-cQ6b46kMIl4L9xommA7oHGtc-DIHMf77OXB2lAQp9XM2o3AXzJuZ_du2plm6Dzxz8_KXXgbhTyyQDHdsR4w0jH2u7ClMaPt6bSKlabweaGCC3Lz7y_56HQw0LF12C6m3vEW9vkV9iB7fxFmi9TjEVnnFVPkiCZa0OHUU-L2iKjipfRecz4O3X3IgF_tykVuFBtsWVz_0TyLPMSMGUMtB-yEXXhIQcTlftL4Q2fS7ToMPWZDZmF5OX9pwHdYvk-1A_BOglClad1RaD0HooNUf8Qr_kScxwxU4TcIIIjQffvAcX9jC3lB_x5tdosfPTQlkiRgqfgBWk73ryvcUImWpV00hdksMFBxW-o8-5leTseFYXDGK_aD_YMIminTUzt602evVSieYTRG1w5VwDJGv_iEcVjMB3zE7SVQYz9vcoMPjlNtmuxL_VCvvPPTmb8OjDha-NMiMGu6jQve4i-5aJNMdeZt-idvGPZjNb81yyZa9CKqt9s9R5YyfO8nlPyAa9c-eWhJp5UWVLlesN2IfWlMcnwCNN0dCfoMCmHfhGrWX5EU-yVFXmHA'
# pylint: enable=line-too-long

ctx = okera.context()
def connect(server, port=12050, token=None):
  if token:
    ctx.enable_token_auth(token_str=token)
  return ctx.connect(host=server, port=port)

def connect_nightly(token=NIGHTLY_TOKEN):
  return connect(server='ec2-34-215-143-132.us-west-2.compute.amazonaws.com', token=token)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: view.py <name of dataset or query>")
    sys.exit(1)

  with connect_nightly() as conn:
    data = conn.scan_as_pandas(sys.argv[1], max_records=1).iloc[0][0]
    print(data.decode('utf-8'))
