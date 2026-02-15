# Security Policy

## Supported Versions

The following versions of AMD AI Compute Observatory receive security updates:

| Version | Supported |
|---------|-----------|
| 1.0.x | Yes |
| < 1.0 | No |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it
responsibly.

### Reporting Process

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Send a detailed report to the project maintainers via GitHub Security Advisories
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Any suggested remediation

### Response Timeline

- **Acknowledgment**: Within 48 hours of report receipt
- **Initial Assessment**: Within 5 business days
- **Resolution Timeline**: Communicated based on severity assessment

### Severity Classification

| Severity | Description | Target Resolution |
|----------|-------------|-------------------|
| Critical | Remote code execution, data breach | 24-48 hours |
| High | Privilege escalation, significant data exposure | 7 days |
| Medium | Limited data exposure, denial of service | 30 days |
| Low | Minor information disclosure | Next release |

## Security Best Practices

When deploying AACO in production environments:

### Configuration

- Restrict file system access to required directories only
- Use principle of least privilege for service accounts
- Enable appropriate logging for audit purposes

### Network

- Deploy behind appropriate network controls
- Use TLS for any network communications
- Restrict access to management interfaces

### Monitoring

- Monitor for unexpected resource consumption
- Review logs for anomalous behavior
- Keep dependencies updated

## Dependency Security

AACO uses automated dependency scanning via GitHub Dependabot and pip-audit. Dependencies with
known vulnerabilities are addressed according to the severity classification above.

## Acknowledgments

We appreciate the security research community's efforts to responsibly disclose vulnerabilities.
Contributors who report valid security issues will be acknowledged (with permission) in release
notes.
