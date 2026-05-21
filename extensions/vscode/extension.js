const vscode = require('vscode');
const http = require('http');
function cfg() {
  const c = vscode.workspace.getConfiguration('adam');
  return { host: c.get('host'), port: c.get('port'), inlineCompletion: c.get('inlineCompletion'), maxTokens: c.get('completionMaxTokens'), persona: c.get('persona') };
}
function postJSON(path, body, timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    const { host, port } = cfg();
    const data = JSON.stringify(body);
    const req = http.request({ host, port, path, method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) }, timeout: timeoutMs }, (res) => {
      let buf = '';
      res.on('data', (c) => buf += c);
      res.on('end', () => { try { resolve(JSON.parse(buf)); } catch (e) { resolve({ raw: buf, parse_error: e.message }); } });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(new Error('timeout')); });
    req.write(data); req.end();
  });
}
function activate(context) {
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  status.text = '$(rocket) Adam'; status.command = 'adam.health'; status.tooltip = 'Click for Adam server status'; status.show();
  context.subscriptions.push(status);
  context.subscriptions.push(vscode.commands.registerCommand('adam.ask', async () => {
    const prompt = await vscode.window.showInputBox({ prompt: 'Ask Adam', placeHolder: 'What do you want to know?' });
    if (!prompt) return;
    const out = vscode.window.createOutputChannel('Adam');
    out.show(true); out.appendLine(`> ${prompt}\n`);
    try {
      const r = await postJSON('/chat/stream', { message: prompt }, 180000);
      out.appendLine(r.raw || JSON.stringify(r, null, 2));
    } catch (e) { out.appendLine(`[error] ${e.message}`); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand('adam.complete', async () => {
    const ed = vscode.window.activeTextEditor;
    if (!ed) return;
    const pos = ed.selection.active;
    const prefix = ed.document.getText(new vscode.Range(new vscode.Position(0, 0), pos));
    const suffix = ed.document.getText(new vscode.Range(pos, new vscode.Position(ed.document.lineCount, 0)));
    const lang = ed.document.languageId;
    try {
      const r = await postJSON('/complete', { prefix: prefix.slice(-1500), suffix: suffix.slice(0, 500), language: lang, max_tokens: cfg().maxTokens }, 30000);
      if (r.completion) await ed.edit((e) => e.insert(pos, r.completion));
    } catch (e) { vscode.window.showErrorMessage(`Adam complete: ${e.message}`); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand('adam.build', async () => {
    const ed = vscode.window.activeTextEditor;
    const goal = ed && !ed.selection.isEmpty ? ed.document.getText(ed.selection) : await vscode.window.showInputBox({ prompt: 'Build goal for Adam' });
    if (!goal) return;
    const out = vscode.window.createOutputChannel('Adam Agentic');
    out.show(true); out.appendLine(`Goal: ${goal}\n`);
    try {
      const r = await postJSON('/chat/stream', { message: goal }, 300000);
      out.appendLine(JSON.stringify(r, null, 2));
    } catch (e) { out.appendLine(`[error] ${e.message}`); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand('adam.health', async () => {
    try {
      const res = await new Promise((resolve, reject) => {
        const { host, port } = cfg();
        http.get({ host, port, path: '/health', timeout: 5000 }, (r) => { let b = ''; r.on('data', (c) => b += c); r.on('end', () => resolve(JSON.parse(b))); }).on('error', reject);
      });
      vscode.window.showInformationMessage(`Adam OK — lessons=${res.adam?.lessons_n} skills=${res.skills?.count} tts=${res.voice?.tts_backend}`);
    } catch (e) { vscode.window.showWarningMessage(`Adam server not reachable at ${cfg().host}:${cfg().port} — ${e.message}`); }
  }));
  if (cfg().inlineCompletion) {
    const provider = {
      async provideInlineCompletionItems(doc, pos, ctx, token) {
        if (token.isCancellationRequested) return;
        const prefix = doc.getText(new vscode.Range(new vscode.Position(0, 0), pos));
        const suffix = doc.getText(new vscode.Range(pos, new vscode.Position(doc.lineCount, 0)));
        try {
          const r = await postJSON('/complete', { prefix: prefix.slice(-1500), suffix: suffix.slice(0, 500), language: doc.languageId, max_tokens: cfg().maxTokens }, 8000);
          if (!r || !r.completion) return { items: [] };
          return { items: [new vscode.InlineCompletionItem(r.completion, new vscode.Range(pos, pos))] };
        } catch (e) { return { items: [] }; }
      }
    };
    context.subscriptions.push(vscode.languages.registerInlineCompletionItemProvider({ scheme: 'file' }, provider));
  }
}
function deactivate() {}
module.exports = { activate, deactivate };
