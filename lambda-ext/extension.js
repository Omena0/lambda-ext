const path = require('path');
const vscode = require('vscode');
const { LanguageClient } = require('vscode-languageclient/node');

let client;
let clientDisposable;

function activate(context) {
    // Start the server from the root of lambda-ext, not a subfolder
    const serverModule = path.join(context.extensionPath, 'server.py');
    const serverOptions = {
        command: 'python',
        args: [serverModule],
        options: { cwd: context.extensionPath }
    };
    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'lambda' }],
        outputChannel: vscode.window.createOutputChannel('Lambda Language Server')
    };
    client = new LanguageClient('lambdaLanguageServer', 'Lambda Language Server', serverOptions, clientOptions);
    clientDisposable = client.start();
    context.subscriptions.push(clientDisposable);
}

function deactivate() {
    if (clientDisposable) {
        return clientDisposable.dispose();
    }
    return undefined;
}

module.exports = {
    activate,
    deactivate
};
