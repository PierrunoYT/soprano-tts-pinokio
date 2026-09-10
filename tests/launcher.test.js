const assert = require('node:assert/strict')
const test = require('node:test')
const vm = require('node:vm')
const launcher = require('../pinokio.js')
const torch = require('../torch.js')

function menu(paths = [], running = [], local = {}) {
  return launcher.menu({}, {
    exists: path => paths.includes(path),
    running: path => running.includes(path),
    local: () => local,
  })
}

test('a partially created environment remains installable', async () => {
  for (const paths of [[], ['app/env']]) {
    const entries = await menu(paths)
    assert.equal(entries[0].href, 'install.js')
    assert.equal(entries[0].default, true)
    assert.ok(!entries.some(entry => entry.href === 'start.js'))
  }
  assert.equal((await menu(['app/env/.installed']))[0].href, 'start.js')
})

test('installation readiness is never written for failed validation output', () => {
  const steps = require('../install.js').run
  const ready = steps.at(-1)
  for (const stdout of [
    'warning: dependency conflict',
    'Traceback: ImportError',
    steps.at(-2).params.message,
  ]) {
    assert.equal(vm.runInNewContext(ready.when.slice(2, -2), {input: {stdout}}), false)
  }
  assert.equal(vm.runInNewContext(ready.when.slice(2, -2), {
    input: {stdout: 'Using device: cpu\r\nSOPRANO_INSTALL_OK\r\n'}
  }), true)
})

test('maintenance stays visible even after reset removes the environment', async () => {
  for (const action of ['install', 'reset', 'update', 'link']) {
    for (const paths of [[], ['app/env'], ['app/env/.installed']]) {
      const entries = await menu(paths, [`${action}.js`])
      assert.equal(entries.length, 1)
      assert.equal(entries[0].href, `${action}.js`)
      assert.equal(entries[0].default, true)
    }
  }
})

test('starting shows terminal until the captured URL is ready', async () => {
  const paths = ['app/env/.installed']
  assert.equal((await menu(paths, ['start.js']))[0].href, 'start.js')
  const script = require('../start.js')
  const pattern = script.run[0].params.on[0].event.slice(1, -1)
  const match = `Running on local URL:  http://127.0.0.1:7861`.match(new RegExp(pattern))
  assert.equal(match[1], 'http://127.0.0.1:7861')
  assert.equal(script.run[1].params.url, '{{input.event[1]}}')
  const entries = await menu(paths, ['start.js'], {url: match[1]})
  assert.equal(entries[0].href, match[1])
  assert.equal(entries[0].default, true)
})

test('each supported platform selects exactly one matching torch build', () => {
  for (const platform of ['win32', 'linux', 'darwin']) {
    for (const gpu of ['nvidia', 'amd', undefined]) {
      const selected = torch.run.filter(step => vm.runInNewContext(
        step.when.slice(2, -2), {platform, gpu}
      ))
      assert.equal(selected.length, 1, `${platform}/${gpu}`)
      const commands = [].concat(selected[0].params.message).join('\n')
      assert.ok(!commands.includes('torch-directml'))
      // Reinstall only the platform wheels: reinstalling their dependencies can
      // upgrade Pillow past Gradio's upper bound and break a clean installation.
      assert.ok(!/--reinstall\s/.test(commands))
      if (platform === 'win32' && gpu === 'amd') {
        assert.ok(commands.includes('/whl/cpu'))
      }
      if (platform !== 'darwin' && gpu === 'nvidia') {
        assert.ok(commands.includes('/whl/cu128'))
        assert.ok(commands.includes('--reinstall-package torch'))
      }
    }
  }
})
