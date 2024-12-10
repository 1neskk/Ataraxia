import os
import subprocess
import platform
from pathlib import Path

import utils

VULKAN_SDK = os.environ.get('VULKAN_SDK')
VULKAN_SDK_INSTALLER_WINDOWS_URL = f'https://sdk.lunarg.com/sdk/download/1.3.283.0/windows/VulkanSDK-1.3.283.0-Installer.exe'
VULKAN_SDK_INSTALLER_LINUX_URL = f'https://sdk.lunarg.com/sdk/download/1.3.283.0/linux/vulkansdk-linux-x86_64-1.3.283.0.tar.xz'
VULKAN_SDK_INSTALLER_MAC_URL = f'https://sdk.lunarg.com/sdk/download/1.3.283.0/mac/vulkansdk-macos-1.3.283.0.dmg'
VULKAN_SDK_LOCAL_PATH = 'PathTracingEngine/thirdparty/VulkanSDK'
VULKAN_SDK_EXE_PATH = Path(VULKAN_SDK_LOCAL_PATH).joinpath('VulkanSDK.exe')

CUDA_TOOLKIT_VERSION = "12.5.1"

def run_command(command: str):
    subprocess.run(command, shell=True)

def check_vulkan_sdk() -> bool:
    if not VULKAN_SDK or not Path(VULKAN_SDK).exists():
        print('Vulkan SDK is not installed or the environment variable is not set.')
        install_vulkan_prompt()
        return False
    else:
        print('Vulkan SDK found.')
        return True

def install_vulkan_prompt():
    answer = input('Vulkan SDK is required to build the project. Do you want to install it? (y/n): ')
    if answer == 'y':
        install_vulkan()
    else:
        print('Vulkan SDK installation cancelled.')

def install_vulkan():
    platform_name = platform.system().lower()

    if platform_name == 'windows':
        Path(VULKAN_SDK_LOCAL_PATH).mkdir(parents=True, exist_ok=True)
        print('Downloading Vulkan SDK...')
        utils.download_file(VULKAN_SDK_INSTALLER_WINDOWS_URL, VULKAN_SDK_EXE_PATH)
        print('Vulkan SDK downloaded.')
        print('Installing Vulkan SDK...')
        os.startfile(VULKAN_SDK_EXE_PATH.absolute())
        print('Re-run this script after installation')
    elif platform_name == 'linux':
        print('Downloading Vulkan SDK...')
        run_command(f'wget {VULKAN_SDK_INSTALLER_LINUX_URL} -O {VULKAN_SDK_LOCAL_PATH}/VulkanSDK.tar.xz')
        print('Vulkan SDK downloaded.')
        print('Extracting Vulkan SDK...')
        run_command(f'tar -xf {VULKAN_SDK_LOCAL_PATH}/VulkanSDK.tar.xz -C {VULKAN_SDK_LOCAL_PATH}')
        print('Vulkan SDK extracted.')
        print('Installing Vulkan SDK...')
        run_command(f'cd {VULKAN_SDK_LOCAL_PATH}/1.3.283.0; ./vulkansdk-installer.sh')
        print('Vulkan SDK installed.')
    elif platform_name == 'darwin':
        print('Downloading Vulkan SDK...')
        run_command(f'wget {VULKAN_SDK_INSTALLER_MAC_URL} -O {VULKAN_SDK_LOCAL_PATH}/VulkanSDK.dmg')
        print('Vulkan SDK downloaded.')
        print('Mounting Vulkan SDK...')
        run_command(f'hdiutil attach {VULKAN_SDK_LOCAL_PATH}/VulkanSDK.dmg')
        print('Vulkan SDK mounted.')
        print('Installing Vulkan SDK...')
        run_command(f'cp -r /Volumes/VulkanSDK/VulkanSDK.app /Applications')
        print('Vulkan SDK installed.')
    else:
        print('Unsupported platform.')

def install_cuda_prompt():
    answer = input('CUDA Toolkit is required to build the project. Do you want to install it? (y/n): ')
    if answer == 'y':
        installer_type = input('Please provide the installer type (local or network): ')
        install_cuda_toolkit(installer_type)
    else:
        print('CUDA Toolkit installation cancelled.')

def check_cuda_toolkit() -> bool:
    platform_name = platform.system().lower()

    if platform_name == 'windows':
        cuda_toolkit_path = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA') / CUDA_TOOLKIT_VERSION
        if not cuda_toolkit_path.exists():
            print('CUDA Toolkit is not installed or the version is not 12.5.1.')
            install_cuda_prompt()
            return False
        else:
            print('CUDA Toolkit found.')
            return True
    elif platform_name == 'linux':
        print('CUDA Toolkit installation is not supported on Linux yet.')  # TODO: Add support for Linux
        return False
    else:
        print('Unsupported platform.')
        return False

def install_cuda_toolkit(installer_type: str):
    platform_name = platform.system().lower()

    if platform_name == 'windows':
        cuda_toolkit_dir = Path('PathTracingEngine/thirdparty/CUDA')
        cuda_toolkit_dir.mkdir(exist_ok=True)
        print('Downloading CUDA Toolkit...')
        if installer_type == 'local':
            utils.download_file(f'https://developer.download.nvidia.com/compute/cuda/{CUDA_TOOLKIT_VERSION}/local_installers/cuda_{CUDA_TOOLKIT_VERSION}_555.85_windows.exe', cuda_toolkit_dir / 'CUDA.exe')
            print('CUDA Toolkit downloaded.')
            print('Installing CUDA Toolkit...')
            os.startfile((cuda_toolkit_dir / 'CUDA.exe').absolute())
            print('Re-run this script after installation')
        elif installer_type == 'network':
            utils.download_file(f'https://developer.download.nvidia.com/compute/cuda/{CUDA_TOOLKIT_VERSION}/network_installers/cuda_{CUDA_TOOLKIT_VERSION}_windows_network.exe', cuda_toolkit_dir / 'CUDA.exe')
            print('CUDA Toolkit downloaded.')
            print('Installing CUDA Toolkit...')
            os.startfile((cuda_toolkit_dir / 'CUDA.exe').absolute())
            print('Re-run this script after installation')
        else:
            print('Unsupported installer type.')
    elif platform_name == 'linux':
        print('CUDA Toolkit installation is not supported on Linux yet.')  # TODO: Add support for Linux
    else:
        print('Unsupported platform.')

def main():
    if not check_vulkan_sdk():
        return

    if not check_cuda_toolkit():
        return

if __name__ == "__main__":
    main()
