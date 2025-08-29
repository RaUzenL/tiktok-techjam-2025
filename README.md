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

# 🚀 Git Workflow  

To keep our work organized, each team member will work on **their own branch**.  

---

## 1. Clone the Repository (first time only)  
```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

---

## 2. Checkout Your Branch  
Each member has their own branch (e.g. `alice`, `bob`, `charlie`). Switch to your branch:  

```bash
git checkout alice
```

---

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

---

## 4. Make Changes & Commit  

Edit your files, then stage and commit changes:  

```bash
git add .
git commit -m "Update: improved booking algorithm"
```

---

## 5. Push Your Work  

```bash
git push origin alice
```

---

## 6. Merge to Main (When Ready)  
- Open a **Pull Request (PR)** from your branch → `main`  
- Request at least **1 reviewer**  
- After approval, merge your branch into `main`  

---

## 🔑 Quick Commands Summary  

```bash
git checkout <your-branch>
git pull origin <your-branch>
# make changes
git add .
git commit -m "Message"
git push origin <your-branch>
```
