# AI Use-Case Hub

## Accounts and role-based access

The app requires a sign-in. The first `governance_admin` account is created at startup from `GOVERNAI_ADMIN_USERNAME` and `GOVERNAI_ADMIN_PASSWORD`, supplied as deployment environment variables or Streamlit secrets. Use a password of at least 12 characters. The app does not provide a default account or password.

On first startup, the app creates `users_information.csv` beside `App.py` with 12 example usernames and assigned roles. These sample rows have blank password hashes and cannot sign in until an administrator sets passwords from **User accounts**. The app checks the submitted username and password against this file; only password hashes are stored.

| Column | Contents |
| --- | --- |
| `username` | Normalized account name |
| `password_hash` | Salted PBKDF2-SHA256 hash; blank means the account is pending and cannot sign in |
| `role` | One of the app's seven roles |
| `created_on` | Date the account password was set |

The example accounts are `alex.submitter`, `jamie.submitter`, `casey.owner`, `taylor.owner`, `priya.privacy`, `morgan.legal`, `riley.risk`, `drew.risk`, `sam.board`, `robin.admin`, `blake.submitter`, and `quinn.owner`. The file is created with owner-only permissions and is excluded from Git. Keep it on persistent storage, back it up securely, and do not share or commit it. This file-based account store is intended for a single app instance with a persistent local filesystem; use an identity provider or shared database-backed account store for multi-instance deployments.

### Role permissions

All signed-in roles can view the dashboard, systems register, alerts, system details, and audit history. Other actions are limited as follows:

| Role | Actions |
| --- | --- |
| `submitter` | Register systems; submit change requests; report incidents |
| `model_owner` | Complete lifecycle checklist items; advance stages when gates pass; manage corrective actions; provide an approval when assigned |
| `privacy_reviewer` | Run AI screening; provide an approval when assigned |
| `legal_reviewer` | Run AI screening; provide an approval when assigned |
| `risk_reviewer` | Run AI screening; provide an approval when assigned; close incidents |
| `review_board` | Provide an approval when assigned |
| `governance_admin` | Create accounts; edit or delete system details; run AI screening; close incidents; reopen returned systems; provide an approval when assigned |

Unauthorized edit controls are replaced with an **Access denied** notice. Passwords should be distributed securely by the administrator and users should avoid reusing passwords.