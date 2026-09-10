module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      when: "{{exists('app/env/.installed')}}",
      method: "fs.rm",
      params: {
        path: "app/env/.installed"
      }
    },
    // Create venv and install dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.11",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Select the PyTorch build for this platform after resolving app dependencies.
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip check && python -c \"import app; print('SOPRANO_INSTALL_' + 'OK')\""
      }
    },
    {
      when: "{{input.stdout.includes('SOPRANO_INSTALL_OK')}}",
      method: "fs.write",
      params: {
        path: "app/env/.installed",
        text: "Installation checks passed."
      }
    }
  ]
}
