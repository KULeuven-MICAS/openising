FROM ubuntu:24.04 AS base
COPY requirements.txt ./

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    curl \
    git \
    tar \
    unzip \
    wget \
    zsh \
    vim \
    nano \
    python3-pip && \
    # Enable break-system-packages for non-venv package installs
    python3 -m pip config set global.break-system-packages true
    
WORKDIR /
RUN pip3 install --no-cache-dir -r requirements.txt
# Install Oh-My-Zsh and Autocomplete Plugin
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    echo "source ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc && \
    chsh -s $(which zsh)
# Zsh as default shell
CMD ["zsh"]
