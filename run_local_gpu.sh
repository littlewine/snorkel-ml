# Set & move to home directory

# Activate nvidia gpu
sudo ls
sudo prime-select nvidia
sudo prime-select intel
optirun nvidia-smi


source set_env.sh
cd "$SNORKELHOME"

# Make sure the submodules are installed
git submodule update --init --recursive

# Make sure parser is installed
PARSER="parser/stanford-corenlp-3.6.0.jar"
if [ ! -f "$PARSER" ]; then
    read -p "CoreNLP [default] parser not found- install now?   " yn
    case $yn in
        [Yy]* ) echo "Installing parser..."; ./install-parser.sh;;
        [Nn]* ) ;;
    esac
fi

# source activate snorkel 27
# python ~/tf_gpu_check.py

# Launch jupyter notebook!
echo "Launching Jupyter Notebook..."
optirun jupyter notebook