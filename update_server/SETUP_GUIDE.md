# BTC-AI Update Server Setup Guide

This guide explains how to set up a server to provide updates for the BTC-AI application.

## Requirements

- A web server with HTTPS support (Apache, Nginx, etc.)
- Ability to host static files
- Optionally, a simple backend script to validate update requests

## Directory Structure

Create the following directory structure on your server:

```
/btc-ai-updates/
  ├── api/
  │   └── updates
  ├── downloads/
  │   ├── btc-ai-0.1.0.zip
  │   ├── btc-ai-0.2.0.zip
  │   └── ...
  └── changelogs/
      ├── 0.1.0.md
      ├── 0.2.0.md
      └── ...
```

## Configuration Files

### Update Information File

Create a file at `/btc-ai-updates/api/updates` that contains the update information in JSON format:

```json
{
  "latest_version": "0.2.0",
  "min_required_version": "0.1.0",
  "release_date": "2025-04-15T10:00:00",
  "download_url": "https://your-server.com/btc-ai-updates/downloads/btc-ai-0.2.0.zip",
  "changelog": "- Improved market prediction accuracy\n- Added support for multiple cryptocurrencies\n- Fixed several UI bugs\n- Enhanced logging for better diagnostics",
  "checksum": "0123456789abcdef0123456789abcdef",
  "size_bytes": 1850000,
  "required": false,
  "auto_update": false
}
```

### Update Package

1. Create a ZIP file containing the updated files:
   - Include only files that have changed since the previous version
   - Maintain the same directory structure as in the application
   - Optionally include an `install.py` script for custom installation logic
   - Optionally include an `exclude.txt` file listing files to exclude from update

2. Calculate the SHA-256 checksum of the ZIP file and include it in the update information.

3. Place the ZIP file in the `/btc-ai-updates/downloads/` directory.

## Web Server Configuration

### Apache Configuration Example

```apache
<VirtualHost *:443>
    ServerName updates.your-domain.com
    DocumentRoot /path/to/btc-ai-updates
    
    <Directory /path/to/btc-ai-updates>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    # SSL Configuration
    SSLEngine on
    SSLCertificateFile /path/to/cert.pem
    SSLCertificateKeyFile /path/to/key.pem
    
    # CORS Headers if needed
    Header set Access-Control-Allow-Origin "*"
    Header set Access-Control-Allow-Methods "GET, OPTIONS"
    Header set Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept"
    
    # Set correct MIME types
    AddType application/json .json
    AddType application/zip .zip
</VirtualHost>
```

### Nginx Configuration Example

```nginx
server {
    listen 443 ssl;
    server_name updates.your-domain.com;
    root /path/to/btc-ai-updates;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        add_header Access-Control-Allow-Origin "*";
        add_header Access-Control-Allow-Methods "GET, OPTIONS";
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept";
    }
    
    location /api/updates {
        default_type application/json;
    }
    
    location /downloads/ {
        default_type application/zip;
    }
}
```

## Advanced: Implementing a Backend Service

For more advanced update management, you can implement a backend service that:

1. Authenticates requests from valid BTC-AI installations
2. Tracks update installations and success rates
3. Provides different update channels (stable, beta, etc.)
4. Implements gradual rollout of updates to detect issues early

Example technologies:
- Python with Flask or FastAPI
- Node.js with Express
- PHP with Laravel or Symfony

## Security Considerations

1. Always serve updates over HTTPS
2. Include checksums for all update packages
3. Sign packages with a private key if possible
4. Implement rate limiting to prevent abuse
5. Regularly back up your update server
6. Keep previous versions available for rollbacks

## Testing the Update System

1. Create a test version of the update server
2. Use different version numbers for testing
3. Test the full update process, including rollbacks
4. Verify that updates apply correctly on different operating systems

## Maintenance

1. Keep a changelog for each version
2. Document all changes in detail
3. Maintain old versions for users who need to roll back
4. Monitor update server logs for errors or unusual activity

## Troubleshooting

Common issues and solutions:

1. **Update not detected**: Check file permissions and ensure the update JSON is accessible.
2. **Download failures**: Verify that the ZIP file exists and is accessible.
3. **Checksum verification failures**: Regenerate the checksum and update the JSON file.
4. **Installation failures**: Check the logs for details and ensure the update package structure matches the application structure.

## Additional Resources

- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) - can be used as an alternative update hosting solution
- [AWS S3](https://aws.amazon.com/s3/) - reliable object storage for update packages
- [CloudFlare](https://www.cloudflare.com/) - add CDN for faster update downloads worldwide 