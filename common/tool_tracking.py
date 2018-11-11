import subprocess as sp
import os.path
import zipfile
import logging

LOGGER = logging.getLogger(__name__)


def save_tool_revision(output_dir):
    """Save tool revision for later debug/reproduction.

    Args:
        output_dir (str): full path to the output directory

    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        tool_info_path = os.path.join(output_dir, "tools.txt")
        with open(tool_info_path, "w", newline='\n') as output_file:
            _save_git_status(output_file)
            _save_cuda_version(output_file)
            _save_python_packages(output_file)

        _save_git_diffs(output_dir)
        _save_untracked_files(output_dir)
    except:
        LOGGER.warning("Could not save tool status")


def _save_git_status(output_file):
    """Store git status.

    - Save current shaid and if repo is dirty or not.
      If git is not available, print a warning and save nothing
    - Save shaid of checked out submodules

    """
    git_commit_id = None
    git_origin_master_id = None
    git_index_clean = None
    git_submodule_names, git_submodule_ids = ([], [])
    try:
        git_commit_id = sp.check_output("git rev-parse HEAD",
                                        shell=True, universal_newlines=True)
        git_origin_master_id = sp.check_output("git rev-parse origin/master",
                                               shell=True, universal_newlines=True)
        git_index_clean = not sp.call("git diff-index --quiet HEAD --", shell=True)
        #  See here (http://stackoverflow.com/a/3879077) and here (http://stackoverflow.com/a/2659808)
        #  for info about this command

        git_submodules_output = sp.check_output("git submodule", shell=True, universal_newlines=True)
        git_submodule_names, git_submodule_ids = _parse_git_submodules_info(git_submodules_output)
    except FileNotFoundError:
        LOGGER.warning("Could not save git commit id. Git command not found?")
    except sp.CalledProcessError:
        LOGGER.warning("Could not save git commit id. Not a git repo?")

    print("---- Git status ----\n", file=output_file)
    print("git_commit_id={}".format(git_commit_id), file=output_file)
    print("git_origin_master_id={}".format(git_origin_master_id), file=output_file)

    for name, shaid in zip(git_submodule_names, git_submodule_ids):
        print("{}:commit_id={}\n".format(name, shaid), file=output_file)

    print("git_index_clean={}\n".format(git_index_clean), file=output_file)


def _parse_git_submodules_info(git_submodules_output):
    output_lines = git_submodules_output.splitlines()
    names = [line[1:].split(" ")[1] for line in output_lines]
    sha_ids = [line[1:].split(" ")[0] for line in output_lines]
    return names, sha_ids


def _save_cuda_version(output_file):
    """Store cuda version info to given file.

    Use nvcc --version and parse response

    """
    try:
        nvcc_output = sp.check_output("nvcc --version",
                                      shell=True, universal_newlines=True)
        cuda_version = _parse_nvcc_output(nvcc_output)
    except FileNotFoundError:
        LOGGER.warning("Could not save cuda version. nvcc command not found?")
        cuda_version = None

    print("---- Cuda version ----\n", file=output_file)
    print("cuda_version={}".format(cuda_version), file=output_file)


def _parse_nvcc_output(nvcc_output):
    cuda_version = None
    v_start = nvcc_output.rfind("V")
    if v_start > -1:
        cuda_version = nvcc_output[v_start + 1:]
    return cuda_version


def _save_python_packages(output_file):
    package_list = sp.check_output("pip list --format=columns",
                                   shell=True, universal_newlines=True)

    print("---- Python packages ----\n", file=output_file)
    print(package_list, file=output_file)


def _save_git_diffs(output_dir):
    dirty_diff = sp.check_output("git diff",
                                 shell=True, universal_newlines=True)
    with open(os.path.join(output_dir, "diff_from_commit.patch"), "w", newline='\n') as output_file:
        print(dirty_diff, file=output_file)

    origin_diff = sp.check_output("git diff origin/master",
                                  shell=True, universal_newlines=True)
    with open(os.path.join(output_dir, "diff_from_origin.patch"), "w", newline='\n') as output_file:
        print(origin_diff, file=output_file)


def _save_untracked_files(output_dir):
    untracked_files = sp.check_output("git ls-files --others --exclude-standard",
                                      shell=True, universal_newlines=True).splitlines()
    zipf = zipfile.ZipFile(os.path.join(output_dir, 'untracked_files.zip'), 'w')
    for untracked_file in untracked_files:
        zipf.write(untracked_file)
    zipf.close()