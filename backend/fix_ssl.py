import ssl
import os

# SSL verification band kar do (temporary fix)
ssl._create_default_https_context = ssl._create_unverified_context

# Environment variables set karo
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

print("SSL fix applied!")