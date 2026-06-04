tnr connect osk66h6a << 'EOF'
if [ -d ~/task2 ] && [ -f ~/task2/train_thunder.py ]; then
    cd ~/task2
fi
echo "Current directory: $(pwd)"
python3 -u train_thunder.py
EOF
