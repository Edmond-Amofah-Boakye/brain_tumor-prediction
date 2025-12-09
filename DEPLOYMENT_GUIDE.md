# 🚀 Hugging Face Deployment Guide - NeuroScan AI

This guide will walk you through deploying your Brain Tumor Classification app to Hugging Face Spaces.

---

## 📋 Prerequisites

Before you start, make sure you have:
- ✅ Git installed on your system
- ✅ Git LFS (Large File Storage) installed
- ✅ A Hugging Face account (free): https://huggingface.co/join
- ✅ Your GitHub repository set up
- ✅ Command line / terminal access

---

## 🎯 Overview: How Your App Works on Hugging Face

### **Model Loading Process**
Your app automatically loads the trained model using this flow:

1. **Automatic Discovery** (`app/config.py` → `get_model_path()`):
   ```python
   # Searches for: results/training_run_*/checkpoints/best_model.pth
   # Finds the latest training run automatically
   ```

2. **User Clicks "Load Model"** in sidebar

3. **Model Loads** from: `results/training_run_20251021_175257/checkpoints/best_model.pth`

4. **Predictions Work** using the loaded PyTorch checkpoint

### **Why It Will Work on Hugging Face**
- ✅ Streamlit is natively supported (no Docker needed)
- ✅ Your model file will be included via Git LFS
- ✅ All dependencies are in `requirements.txt`
- ✅ Code uses relative paths (works in any environment)
- ✅ Automatically detects CPU/GPU (HF free tier uses CPU)

---

## 📦 Step 1: Install Git LFS

Your model file is **134 MB**, which exceeds GitHub's 100MB limit. We need Git LFS.

### **Windows:**
```bash
# Download and install from: https://git-lfs.github.com/
# Or using winget:
winget install -e --id GitHub.GitLFS

# Verify installation
git lfs version
```

### **Mac:**
```bash
brew install git-lfs
git lfs version
```

### **Linux:**
```bash
sudo apt-get install git-lfs
git lfs version
```

### **Initialize Git LFS**
```bash
cd c:\Users\User\Desktop\brain_tumor_analysis
git lfs install
```

Expected output: `Git LFS initialized.`

---

## 🔧 Step 2: Configure Git Tracking

### **Add the .gitattributes file** (Already created ✅)
This file tells Git LFS to track `.pth` files:
```bash
git add .gitattributes
```

### **Verify Git LFS is tracking correctly**
```bash
git lfs track
```

Expected output:
```
Listing tracked patterns
    *.pth (filter=lfs diff=lfs merge=lfs -text)
    results/training_run_*/checkpoints/best_model.pth (filter=lfs diff=lfs merge=lfs -text)
```

---

## 📂 Step 3: Add Your Model File

### **Check current git status**
```bash
git status
```

### **Add the model file**
```bash
git add results/training_run_20251021_175257/checkpoints/best_model.pth
```

### **Verify it's being tracked by LFS** (IMPORTANT!)
```bash
git lfs ls-files
```

You should see `best_model.pth` listed. If not, something went wrong with LFS setup.

### **Add other necessary files**
```bash
git add .gitignore
git add .gitattributes
git add README_HUGGINGFACE.md
git add requirements.txt
git add app/
git add models/
git add explainability/
git add data/data_loader.py
git add results/training_run_20251021_175257/config.json
git add results/training_run_20251021_175257/test_results.json
```

### **Commit your changes**
```bash
git commit -m "Add trained model and prepare for Hugging Face deployment"
```

---

## ☁️ Step 4: Push to GitHub

### **Push to your repository**
```bash
git push origin main
```

**Note**: The first push with LFS might take a while (uploading 134MB model).

### **Verify on GitHub**
1. Go to your GitHub repository
2. Navigate to `results/training_run_20251021_175257/checkpoints/`
3. You should see `best_model.pth` with an "LFS" badge
4. File size should show as ~134 MB

---

## 🤗 Step 5: Create Hugging Face Space

### **Option A: Link from GitHub (RECOMMENDED)**

1. **Go to Hugging Face**: https://huggingface.co/new-space

2. **Fill in the form**:
   - **Owner**: Your username
   - **Space name**: `neuroscan-ai` (or your choice)
   - **License**: MIT
   - **Select SDK**: Streamlit
   - **Space hardware**: CPU basic (free tier)

3. **Link GitHub Repository**:
   - Click "Link a GitHub repository"
   - Authorize Hugging Face to access your repos
   - Select: `Edmond-Amofah-Boakye/brain_tumor-prediction`

4. **Configure Space**:
   - The system will detect your `README_HUGGINGFACE.md`
   - Make sure it has the YAML frontmatter at the top

5. **Deploy!**
   - Click "Create Space"
   - Hugging Face will automatically:
     - Clone your repo
     - Download the LFS model file
     - Install dependencies from `requirements.txt`
     - Run `streamlit run app/main.py`

### **Option B: Direct Upload (Alternative)**

If linking doesn't work, you can push directly to HF:

```bash
# Clone the HF space
git clone https://huggingface.co/spaces/YOUR_USERNAME/neuroscan-ai
cd neuroscan-ai

# Copy your files
cp -r path/to/brain_tumor_analysis/* .

# Rename README
mv README_HUGGINGFACE.md README.md

# Push
git add .
git commit -m "Initial deployment"
git push
```

---

## 🎯 Step 6: Configure the Hugging Face Space

### **Rename README (if using Option A)**

After the space is created:

1. Go to Files tab in your HF Space
2. Delete or rename existing `README.md`
3. Rename `README_HUGGINGFACE.md` → `README.md`
4. The YAML frontmatter MUST be at the very top of `README.md`

### **Verify Configuration**

Your `README.md` should start with:
```yaml
---
title: NeuroScan AI - Brain Tumor Classifier
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app/main.py
pinned: false
license: mit
---
```

---

## 🔍 Step 7: Verify Deployment

### **Check Build Logs**
1. Go to your Space page
2. Click on "Logs" or "Build" tab
3. Watch for:
   - ✅ Dependencies installing
   - ✅ Model file downloading (LFS)
   - ✅ Streamlit starting
   - ❌ Any error messages

### **Common Build Messages**
```
Installing dependencies from requirements.txt...
Downloading LFS files...
Starting Streamlit application...
✓ App is running on port 7860
```

### **Test Your App**
1. Once build completes, the app should load automatically
2. **Click "🔄 Load Model"** in the sidebar (IMPORTANT!)
3. Upload a test brain scan image
4. Click "🔍 ANALYZE IMAGE"
5. Verify predictions appear

---

## 🐛 Troubleshooting

### **Issue 1: Model File Not Found**
**Symptoms**: App loads but clicking "Load Model" fails

**Solutions**:
1. Verify LFS file was downloaded:
   ```bash
   # In HF Space terminal (if available)
   ls -lh results/training_run_20251021_175257/checkpoints/
   ```
2. Check file isn't a pointer (should be ~134MB, not ~130 bytes)
3. Re-push with LFS: `git lfs push --all origin main`

### **Issue 2: Dependencies Not Installing**
**Symptoms**: Build fails with package errors

**Solutions**:
1. Check `requirements.txt` has all packages
2. Pin specific versions if needed:
   ```
   torch==1.12.0
   streamlit==1.28.0
   ```
3. Check Python version compatibility (HF uses Python 3.8+)

### **Issue 3: Out of Memory**
**Symptoms**: App crashes when loading model or making predictions

**Solutions**:
1. Your model loads to CPU automatically ✅
2. If still issues, upgrade Space hardware (paid)
3. Consider model quantization (advanced)

### **Issue 4: Streamlit Not Starting**
**Symptoms**: Build succeeds but app doesn't load

**Solutions**:
1. Verify `app_file: app/main.py` in README.md
2. Check app/main.py has correct entry point:
   ```python
   if __name__ == "__main__":
       main()
   ```
3. Test locally: `streamlit run app/main.py`

### **Issue 5: GradCAM/Visualization Errors**
**Symptoms**: Predictions work but explanations fail

**Solutions**:
1. Check all visualization dependencies installed
2. Verify captum package: `pip install captum`
3. Review error logs for specific missing packages

---

## 📊 Post-Deployment Checklist

After successful deployment:

- [ ] App loads without errors
- [ ] "Load Model" button works
- [ ] Can upload images
- [ ] Predictions display correctly
- [ ] Confidence scores show properly
- [ ] Symmetry analysis works
- [ ] GradCAM heatmaps generate
- [ ] Clinical reports download
- [ ] All 4 tabs function (Predictions, Symmetry, Explanations, Report)

---

## 🔄 Updating Your Deployed App

When you make changes:

```bash
# Make your changes locally
# Test locally: streamlit run app/main.py

# Commit and push
git add .
git commit -m "Update: [describe changes]"
git push origin main

# If Space is linked to GitHub, it auto-updates!
# Otherwise, push to HF Space manually
```

---

## 📈 Monitoring Your Space

### **View Analytics**
- Go to your Space → "Analytics" tab
- See usage statistics, unique users, etc.

### **Check Logs**
- Go to "Logs" tab to see real-time application logs
- Useful for debugging user-reported issues

### **Manage Hardware**
- Free tier: CPU basic (2 vCPU, 16GB RAM)
- Upgrade options available for faster inference

---

## 🎓 Best Practices

1. **Keep README Updated**: The README.md is the Space homepage
2. **Add Example Images**: Include sample brain scans in README
3. **Monitor Performance**: Check logs regularly
4. **Version Control**: Tag releases in git
5. **User Feedback**: Enable discussions in Space settings
6. **Medical Disclaimer**: Keep prominent in UI and README ⚠️

---

## 🔗 Useful Links

- **Hugging Face Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Streamlit on HF**: https://huggingface.co/docs/hub/spaces-sdks-streamlit
- **Git LFS**: https://git-lfs.github.com/
- **Your GitHub Repo**: https://github.com/Edmond-Amofah-Boakye/brain_tumor-prediction

---

## 📞 Need Help?

If you encounter issues:

1. **Check HF Community**: https://discuss.huggingface.co/
2. **Streamlit Forums**: https://discuss.streamlit.io/
3. **GitHub Issues**: Open an issue in your repo
4. **Space Discussion**: Enable and use discussion tab on your Space

---

## 🎉 Success!

Once deployed, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/neuroscan-ai
```

Share this link with:
- 🏥 Medical professionals for testing
- 🎓 Researchers for collaboration  
- 📚 Students for educational purposes
- 🌟 Community for feedback

---

**Remember**: This is a research/educational tool. Always include medical disclaimers and encourage professional medical consultation! ⚠️

**Good luck with your deployment! 🚀**
