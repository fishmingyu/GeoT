
INSTALL_PATH="/home/"
if [ "${BUILD_PLATFORM}" == "x86_64" ]; then
    Anaconda="Anaconda3-2023.03-1-Linux-x86_64.sh"
else
    echo "Unsupported platform: '${BUILD_PLATFORM}'"
	exit 1
fi

cd ${INISTALL_PATH}
curl -O https://repo.anaconda.com/archive/${Anaconda}
chmod 777 ${Anaconda}
bash ${Anaconda} -b -p /usr/local/anaconda3
rm ${Anaconda}
echo "export PATH="/usr/local/anaconda3/bin:$PATH"" > ~/.bashrc
source ~/.bashrc
