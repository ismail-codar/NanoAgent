import docker
import os
from typing import Optional



class DockerSandbox:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def create_container(self):
        try:
            image, build_logs = self.client.images.build(
                path=os.path.dirname(os.path.abspath(__file__)),
                tag="agent-sandbox",
                rm=True,
                forcerm=True,
                buildargs={},
                # decode=True
            )
        except docker.errors.BuildError as e:
            print("Build error logs:")
            for log in e.build_log:
                if 'stream' in log:
                    print(log['stream'].strip())
            raise

        # Create container with security constraints and proper logging
        self.container = self.client.containers.run(
            "agent-sandbox",
            name="python-sandbox",
            command="tail -f /dev/null",  # Keep container running
            detach=True,
            tty=True,
            mem_limit="64m",
            cpu_quota=50000,
            pids_limit=100,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            environment={
                "HF_TOKEN": os.getenv("HF_TOKEN")
            },
        )

    def run_code(self, code: str) -> Optional[str]:
        if not self.container:
            self.create_container()

        # Execute code in container
        exec_result = self.container.exec_run(
            cmd=["python", "-c", code],
            user="nobody"
        )

        # Collect all output
        return exec_result.output.decode() if exec_result.output else None


    def cleanup(self):
        if self.container:
            try:
                self.container.stop()
            except docker.errors.NotFound:
                # Container already removed, this is expected
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.container = None  # Clear the reference

# Reference: https://huggingface.co/docs/smolagents/tutorials/secure_code_execution#sandbox-approaches-for-secure-code-execution
if __name__ == '__main__':
    # Example usage:

    print(os.path.dirname(os.path.abspath(__file__)))
    sandbox = DockerSandbox()

    try:
        # Define your agent code
        agent_code = """print('Hello world')"""
        for i in range(10):
            # Run the code in the sandbox
            output = sandbox.run_code(agent_code)
            print(output)

    finally:
        sandbox.cleanup()