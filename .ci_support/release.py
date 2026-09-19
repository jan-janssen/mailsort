import tomllib


def get_exact_pin_versions(pyproject_content):
    pyproject_dict = tomllib.loads(pyproject_content)
    versions = {}
    for section, key in [("build-system", "requires"), ("project", "dependencies")]:
        for dep in pyproject_dict.get(section, {}).get(key, []):
            if "==" in dep:
                name, version = dep.split("==", 1)
                versions[name] = version
    return versions


def get_env_versions(env_content):
    versions = {}
    in_dependencies = False
    for raw_line in env_content.splitlines():
        line = raw_line.strip()
        if line == "dependencies:":
            in_dependencies = True
            continue
        if not in_dependencies:
            continue
        if raw_line and not raw_line.startswith((" ", "\t", "-")):
            break
        if not line.startswith("-"):
            continue
        dep = line.lstrip("-").strip()
        if "=" in dep:
            name, version = dep.split("=", 1)
            if name and version:
                versions[name.strip()] = version.strip()
    return versions


def to_release_constraint(dep, high_version, low_version):
    if low_version == high_version:
        return f"{dep}=={high_version}"
    return f"{dep}>={low_version},<={high_version}"


def update_dependencies(pyproject_content, version_low_dict, version_high_dict):
    missing_dependencies = sorted(
        [dep for dep in version_high_dict if dep not in version_low_dict]
    )
    if missing_dependencies:
        msg = "Missing lower-bound versions for dependencies: " + ", ".join(
            missing_dependencies
        )
        raise ValueError(msg)

    updated_content = pyproject_content
    for dep, high_version in version_high_dict.items():
        old = f"{dep}=={high_version}"
        new = to_release_constraint(dep, high_version, version_low_dict[dep])
        updated_content = updated_content.replace(old, new)
    return updated_content


def update_pyproject_for_release(pyproject_content, env_content):
    version_high_dict = get_exact_pin_versions(pyproject_content=pyproject_content)
    version_low_dict = get_env_versions(env_content=env_content)
    return update_dependencies(
        pyproject_content=pyproject_content,
        version_low_dict=version_low_dict,
        version_high_dict=version_high_dict,
    )


if __name__ == "__main__":
    with open("pyproject.toml", "r", encoding="utf-8") as file:
        setup_content = file.read()

    with open("environment.yml", "r", encoding="utf-8") as file:
        env_content = file.read()

    setup_content_new = update_pyproject_for_release(
        pyproject_content=setup_content, env_content=env_content
    )

    with open("pyproject.toml", "w", encoding="utf-8") as file:
        file.write(setup_content_new)
