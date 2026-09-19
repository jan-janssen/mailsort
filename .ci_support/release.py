def get_setup_version_and_pattern(setup_content):
    depend_lst, version_lst = [], []
    for line in setup_content:
        if "==" in line:
            dep_lst = (
                line.split("= [")[-1]
                .split("]\n")[0]
                .replace(" ", "")
                .replace('"', "")
                .replace("'", "")
                .split(",")
            )
            for dep in dep_lst:
                if dep != "\n":
                    version_lst.append(dep.split("==")[1])
                    depend_lst.append(dep.split("==")[0])
    return {dep: version for dep, version in zip(depend_lst, version_lst)}


def get_env_version(env_content):
    read_flag = False
    depend_lst, version_lst = [], []
    for line in env_content:
        if "dependencies:" in line:
            read_flag = True
        elif read_flag:
            dep_lst = line.replace("-", "").replace(" ", "").replace("\n", "").split("=")
            if len(dep_lst) == 2:
                depend_lst.append(dep_lst[0])
                version_lst.append(dep_lst[1])
    return {dep: version for dep, version in zip(depend_lst, version_lst)}


def update_dependencies(setup_content, version_low_dict, version_high_dict):
    version_combo_dict = {}
    for dep, ver in version_high_dict.items():
        if dep in version_low_dict and version_low_dict[dep] != ver:
            version_combo_dict[dep] = f"{dep}>={version_low_dict[dep]},<={ver}"
        else:
            version_combo_dict[dep] = f"{dep}=={ver}"

    setup_content_new = ""
    pattern_dict = {dep: f"{dep}=={ver}" for dep, ver in version_high_dict.items()}
    for line in setup_content:
        for dep, pattern in pattern_dict.items():
            if pattern in line:
                line = line.replace(pattern, version_combo_dict[dep])
        setup_content_new += line
    return setup_content_new


if __name__ == "__main__":
    with open("pyproject.toml", "r", encoding="utf-8") as file:
        setup_content = file.readlines()

    with open("environment.yml", "r", encoding="utf-8") as file:
        env_content = file.readlines()

    setup_content_new = update_dependencies(
        setup_content=setup_content[2:],
        version_low_dict=get_env_version(env_content=env_content),
        version_high_dict=get_setup_version_and_pattern(setup_content=setup_content[2:]),
    )

    with open("pyproject.toml", "w", encoding="utf-8") as file:
        file.writelines("".join(setup_content[:2]) + setup_content_new)
