# Managing Code Owners and Required Reviews

To ensure GitHub requires your approval on every pull request that touches files with a designated code owner, complete both steps below.

1. **Define the code owner in the repository**  
   The repository already contains `.github/CODEOWNERS` with the following rule:
   ```
   * @kv-ops
   ```
   This makes `@kv-ops` the default owner for the entire repository.

2. **Enable the branch protection rule that enforces code-owner reviews**
   In the GitHub UI, go to **Settings → Branches → Branch protection rules** and edit (or create) the rule for your default branch (for example `main`). Enable the following checkboxes:
   - **Require a pull request before merging**
   - **Require review from Code Owners**

   Optional but recommended:
   - Keep **Require approvals** enabled and set the approval count to `1`
   - Disable "Allow bypassing the above settings" for everyone except administrators who should be able to override the rule

Once these settings are saved, GitHub will block merges until `@kv-ops` has approved any pull request that modifies iles covered by the CODEOWNERS rule. No additional files or configuration changes inside the repository are needed beyond `.github/CODEOWNERS`.
