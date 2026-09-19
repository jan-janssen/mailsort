import importlib.util
import pathlib
import unittest


def _load_release_module():
    file_path = (
        pathlib.Path(__file__).resolve().parent.parent / ".ci_support" / "release.py"
    )
    spec = importlib.util.spec_from_file_location("release", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReleaseDependencyUpdate(unittest.TestCase):
    def setUp(self):
        self.release = _load_release_module()

    def test_dependency_range_when_versions_differ(self):
        pyproject = """
[build-system]
requires = ["hatchling==1.32.0"]

[project]
dependencies = ["numpy==2.5.1"]
"""
        environment = """
channels:
- conda-forge
dependencies:
- numpy =1.23.5
- hatchling =1.32.0
"""
        updated = self.release.update_pyproject_for_release(
            pyproject_content=pyproject, env_content=environment
        )
        self.assertIn("numpy>=1.23.5,<=2.5.1", updated)

    def test_dependency_keeps_exact_pin_when_versions_match(self):
        pyproject = """
[build-system]
requires = ["hatchling==1.32.0"]

[project]
dependencies = ["numpy==2.5.1"]
"""
        environment = """
dependencies:
- numpy =2.5.1
- hatchling =1.32.0
"""
        updated = self.release.update_pyproject_for_release(
            pyproject_content=pyproject, env_content=environment
        )
        self.assertIn("numpy==2.5.1", updated)

    def test_raises_when_lower_bound_missing(self):
        pyproject = """
[project]
dependencies = ["numpy==2.5.1"]
"""
        environment = """
dependencies:
- pandas =1.5.3
"""
        with self.assertRaises(ValueError):
            self.release.update_pyproject_for_release(
                pyproject_content=pyproject, env_content=environment
            )

    def test_strict_greater_than_is_preserved(self):
        pyproject = """
[project]
dependencies = ["numpy==2.5.1"]
"""
        environment = """
dependencies:
- numpy >1.23.5
"""
        updated = self.release.update_pyproject_for_release(
            pyproject_content=pyproject, env_content=environment
        )
        self.assertIn("numpy>1.23.5,<=2.5.1", updated)

    def test_nested_pip_dependencies_raise_error(self):
        pyproject = """
[project]
dependencies = ["numpy==2.5.1"]
"""
        environment = """
dependencies:
- pip:
  - numpy==1.23.5
"""
        with self.assertRaises(ValueError):
            self.release.update_pyproject_for_release(
                pyproject_content=pyproject, env_content=environment
            )

    def test_conda_build_string_is_ignored(self):
        pyproject = """
[project]
dependencies = ["numpy==2.5.1"]
"""
        environment = """
dependencies:
- numpy =1.23.5 py310_0
"""
        updated = self.release.update_pyproject_for_release(
            pyproject_content=pyproject, env_content=environment
        )
        self.assertIn("numpy>=1.23.5,<=2.5.1", updated)


if __name__ == "__main__":
    unittest.main()
