# tiktok-techjam-2025
## Meeting Highlights:
1. more data comment: from crawling (manual labelling, negative review)
2. metadata: avail or not, literature review -> review history/time
3. data collection -> unified format
4. Multi-agent
5. model: api (deepseek, hugging face
6. Interactive demo: only sample data needed
## Task Division:
1. Data part: 3 
2. Agent/model: 2
3. Deliverable for submission: sat night to discuss

## Submission Requirement
1. Build or update a working solution that addresses one of the 7 problem statements
2. Include a text description that should explain the features and functionality of the project, and also include:
  - Development tools used to build the project
  - APIs used in the project
  - Assets used in the project
  - Libraries used in the project
  - The relevant problem statement 
3. Include a link to the team's public Github repository with Readme
4. Include a demonstration video of the project. The video portion of the submission:
  - should be less than three (3) minutes
  - should include footage that shows the project functioning on the device for which it was built
  - must be uploaded to and made publicly visible on YouTube, and a link to the video must be provided in the text description; and
  - must not include third party trademarks, or copyrighted music or other material unless the Entrant has permission to use such material.
5. Ensure that all deliverables detailed in the problem statement are addressed.

# 🚀 Git Workflow  

To keep our work organized, each team member will work on **their own branch**.  


## 1. Clone the Repository (first time only)  
```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```


## 2. Checkout Your Branch  
Each member has their own branch (e.g. `alice`, `bob`, `charlie`). 
Add ur branch:

```bash
git branch <branch-name>
```


Switch to your branch:  

```bash
git checkout alice
```


## 3. Keep Your Branch Updated  

**Before working, pull the latest changes from your branch:**  

```bash
git pull origin alice
```

**If you need the latest `main` updates in your branch:**  

```bash
git checkout main
git pull origin main
git checkout alice
git merge main   # or rebase if you prefer
```


## 4. Make Changes & Commit  

Edit your files, then stage and commit changes:  

```bash
git add .
git commit -m "Update: improved booking algorithm"
```


## 5. Push Your Work  

```bash
git push origin alice
```


## 6. Merge to Main (When Ready)  
- Open a **Pull Request (PR)** from your branch → `main`  
- Request at least **1 reviewer**  
- After approval, merge your branch into `main`  


## 🔑 Quick Commands Summary  

```bash
git checkout <your-branch>
git pull origin <your-branch>
# make changes
git add .
git commit -m "Message"
git push origin <your-branch>
```
