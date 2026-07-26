# Publishing this repository anonymously

This folder is already a local Git repository with neutral commit metadata.

## 1. Create a fresh GitHub account

Use a new email address and a username that cannot be connected to any author.
Do not add a real name, institutional affiliation, personal avatar, biography,
recovery email visible to collaborators, or links to personal profiles.

## 2. Create an empty private or public repository

Do not initialize it with a README, license, or `.gitignore`, since those files
are already included here.

## 3. Push this local repository

Replace the placeholder URL below by the URL of the new repository:

    git remote add origin https://github.com/ANONYMOUS-ACCOUNT/REPOSITORY.git
    git branch -M main
    git push -u origin main

## 4. Verify anonymity

Check:

    git log --format=fuller

Also inspect the GitHub repository page in a private browser window. Verify that
the account name, commit author, email, notebook metadata, repository
description, issues, and linked activity do not reveal the authors.

## Important

GitHub itself will know account and network metadata. “Anonymous” here means
anonymous to reviewers and the public, not anonymous to GitHub.
