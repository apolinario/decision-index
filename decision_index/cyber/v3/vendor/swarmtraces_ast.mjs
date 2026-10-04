/** Read-only static analysis of complete JavaScript artifacts. Never runs input. */
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { createInterface } from 'node:readline';

const require = createRequire(import.meta.url);
const ts = require(process.argv[2]);
if (ts.version !== '5.9.3') throw new Error('Expected TypeScript 5.9.3 parser');
const credentialHeaders = new Set(['authorization', 'x-api-key', 'api-key', 'x-auth-token', 'x-api-token']);
const globals = new Set(['fetch', 'XMLHttpRequest', 'document', 'console', 'JSON', 'window', 'atob', 'btoa',
  'Promise', 'URL', 'Buffer', 'TextEncoder', 'TextDecoder', 'String', 'Object', 'Array', 'Math']);
const bodyMethods = new Set(['slice', 'substring', 'substr', 'replace', 'replaceAll', 'toString', 'indexOf', 'match', 'split', 'join']);
const hash = value => createHash('sha256').update(value).digest('hex');

function walk(node, visit) {
  visit(node);
  ts.forEachChild(node, child => walk(child, visit));
}

function propertyName(node) {
  return node && (ts.isIdentifier(node) || ts.isStringLiteral(node)) ? node.text : null;
}

function member(node, name) {
  return ts.isPropertyAccessExpression(node) && node.name.text === name;
}

function depends(node, name) {
  let yes = false;
  walk(node, child => {
    if (ts.isIdentifier(child) && child.text === name
      && !(ts.isPropertyAccessExpression(child.parent) && child.parent.name === child)) yes = true;
  });
  return yes;
}

function changedNames(node) {
  const names = new Set();
  walk(node, child => { if (ts.isIdentifier(child)) names.add(child.text); });
  return names;
}

function cleanCallback(callback, builtin) {
  const name = callback.parameters[0]?.name;
  if (!name || !ts.isIdentifier(name) || callback.parameters.length !== 1
    || callback.parameters[0].initializer || callback.parameters[0].dotDotDotToken) return false;
  let clean = true;
  walk(callback.body, node => {
    if (ts.isFunctionLike(node) || ts.isIfStatement(node) || ts.isConditionalExpression(node)
      || ts.isSwitchStatement(node) || ts.isIterationStatement(node, false) || ts.isTryStatement(node)
      || ts.isThrowStatement(node)) clean = false;
    if (ts.isReturnStatement(node)
      && (!ts.isBlock(callback.body) || node !== callback.body.statements.at(-1))) clean = false;
    if (ts.isBinaryExpression(node) && node.operatorToken.kind >= ts.SyntaxKind.FirstAssignment
      && node.operatorToken.kind <= ts.SyntaxKind.LastAssignment && depends(node.left, name.text)) clean = false;
    if ((ts.isPrefixUnaryExpression(node) || ts.isPostfixUnaryExpression(node))
      && [ts.SyntaxKind.PlusPlusToken, ts.SyntaxKind.MinusMinusToken].includes(node.operator)
      && depends(node.operand, name.text)) clean = false;
    if (ts.isVariableDeclaration(node) && changedNames(node.name).has(name.text)) clean = false;
    if (ts.isDeleteExpression(node) && depends(node.expression, name.text)) clean = false;
    if (ts.isCallExpression(node) && depends(node, name.text)) {
      const fn = node.expression;
      const direct = ts.isIdentifier(fn) && builtin(fn.text)
        && ['fetch', 'eval', 'atob', 'btoa', 'encodeURIComponent', 'decodeURIComponent'].includes(fn.text);
      const property = ts.isPropertyAccessExpression(fn)
        && ((bodyMethods.has(fn.name.text) && depends(fn.expression, name.text))
          || (ts.isIdentifier(fn.expression) && builtin(fn.expression.text)
            && ({ JSON: ['stringify'], document: ['write', 'writeln'], console: ['log', 'info', 'error'] })[fn.expression.text]?.includes(fn.name.text)));
      if (!direct && !property) clean = false;
      if (ts.isIdentifier(fn) && fn.text === 'eval') {
        const last = ts.isBlock(callback.body) ? callback.body.statements.at(-1) : null;
        if (node !== callback.body && node !== last?.expression) clean = false;
      }
    }
  });
  return clean;
}

function bodyReader(node) {
  if (!node || !ts.isArrowFunction(node) || node.parameters.length !== 1
    || !ts.isIdentifier(node.parameters[0].name) || !ts.isCallExpression(node.body)
    || node.body.arguments.length !== 0) return false;
  const fn = node.body.expression;
  return ts.isPropertyAccessExpression(fn) && ['text', 'json'].includes(fn.name.text)
    && ts.isIdentifier(fn.expression) && fn.expression.text === node.parameters[0].name.text;
}

function fingerprint(node) {
  if (ts.isIdentifier(node)) {
    const p = node.parent;
    const key = (ts.isPropertyAccessExpression(p) && p.name === node)
      || ((ts.isPropertyAssignment(p) || ts.isMethodDeclaration(p)) && p.name === node);
    return ['Identifier', key || globals.has(node.text) ? node.text : '_'];
  }
  if (ts.isStringLiteralLike(node)) {
    const p = node.parent;
    return ['String', ts.isPropertyAssignment(p) && p.name === node ? node.text : '_'];
  }
  if (ts.isNumericLiteral(node)) return ['Number'];
  const children = [];
  ts.forEachChild(node, child => { children.push(fingerprint(child)); });
  return [ts.SyntaxKind[node.kind], ...children];
}

function analyze(text) {
  const file = ts.createSourceFile('artifact.js', text, ts.ScriptTarget.ESNext, true, ts.ScriptKind.JS);
  if (file.parseDiagnostics.length) return { admitted: false, reason: 'parse_error' };
  const bindings = new Map(), bindingScopes = new Map(), writes = new Map(), calls = [];
  const unsafeWrites = new Set(), mutatedBuiltins = new Set();
  function scope(node) {
    let parent = node.parent;
    while (parent && !ts.isBlock(parent) && !ts.isSourceFile(parent)) parent = parent.parent;
    return parent ?? file;
  }
  walk(file, node => {
    if ((ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node)) && node.name) {
      writes.set(node.name.text, (writes.get(node.name.text) || 0) + 2);
    }
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name)) {
      const name = node.name.text;
      writes.set(name, (writes.get(name) || 0) + 1);
      if (node.initializer) {
        bindings.set(name, node.initializer);
        bindingScopes.set(name, scope(node));
        if (!ts.isVariableStatement(node.parent.parent)
          || node.parent.parent.parent !== scope(node)) unsafeWrites.add(name);
      }
    }
    if ((ts.isVariableDeclaration(node) || ts.isParameter(node)) && !ts.isIdentifier(node.name)) {
      for (const name of changedNames(node.name)) unsafeWrites.add(name);
    }
    if (ts.isForOfStatement(node) || ts.isForInStatement(node)) {
      for (const name of changedNames(node.initializer)) unsafeWrites.add(name);
    }
    if (ts.isParameter(node) && ts.isIdentifier(node.name)) writes.set(node.name.text, (writes.get(node.name.text) || 0) + 2);
    if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.EqualsToken && ts.isIdentifier(node.left)) {
      const name = node.left.text;
      writes.set(name, (writes.get(name) || 0) + 1);
      bindings.set(name, node.right);
      bindingScopes.set(name, scope(node));
      if (!ts.isExpressionStatement(node.parent) || node.parent.parent !== scope(node)) unsafeWrites.add(name);
    }
    if (ts.isBinaryExpression(node) && node.operatorToken.kind >= ts.SyntaxKind.FirstAssignment
      && node.operatorToken.kind <= ts.SyntaxKind.LastAssignment
      && (node.operatorToken.kind !== ts.SyntaxKind.EqualsToken || !ts.isIdentifier(node.left))) {
      for (const name of changedNames(node.left)) unsafeWrites.add(name);
      // Writing a DOM output is the sink itself, not a replacement of the
      // document API. Other member writes invalidate built-in assumptions.
      const domOutput = node.operatorToken.kind === ts.SyntaxKind.EqualsToken
        && ts.isPropertyAccessExpression(node.left)
        && ['innerText', 'textContent', 'innerHTML'].includes(node.left.name.text);
      for (const name of changedNames(node.left)) {
        if (name !== 'document' || !domOutput) mutatedBuiltins.add(name);
      }
    }
    if ((ts.isPrefixUnaryExpression(node) || ts.isPostfixUnaryExpression(node))
      && [ts.SyntaxKind.PlusPlusToken, ts.SyntaxKind.MinusMinusToken].includes(node.operator)) {
      for (const name of changedNames(node.operand)) unsafeWrites.add(name);
      for (const name of changedNames(node.operand)) mutatedBuiltins.add(name);
    }
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'fetch') calls.push(node);
  });
  const template = hash(JSON.stringify(fingerprint(file)));
  if (writes.has('fetch') || unsafeWrites.has('fetch')) return { admitted: false, reason: 'shadowed_fetch', template };
  const builtin = name => !writes.has(name) && !mutatedBuiltins.has(name);

  function resolve(node, before, seen = new Set()) {
    if (!node) return null;
    if (ts.isParenthesizedExpression(node)) return resolve(node.expression, before, seen);
    if (!ts.isIdentifier(node)) return node;
    if (seen.has(node.text) || unsafeWrites.has(node.text) || writes.get(node.text) !== 1) return null;
    const value = bindings.get(node.text);
    if (!value || value.end >= before) return null;
    const visibleScope = bindingScopes.get(node.text);
    if (!visibleScope || before < visibleScope.pos || before >= visibleScope.end) return null;
    return resolve(value, before, new Set([...seen, node.text]));
  }

  function object(node, before) {
    // Object aliases may have been mutated. Admit inline options/header objects
    // only until mutation-aware analysis exists.
    if (!node || !ts.isObjectLiteralExpression(node)) return null;
    const value = node;
    if (!value || !ts.isObjectLiteralExpression(value)) return null;
    const fields = new Map();
    for (const prop of value.properties) {
      if (!ts.isPropertyAssignment(prop) || !propertyName(prop.name)) return null;
      const key = propertyName(prop.name);
      if (fields.has(key)) return null;
      fields.set(key, prop.initializer);
    }
    return fields;
  }

  function fromResponse(node, call) {
    let parent = call.parent;
    while (parent) {
      if ((ts.isArrowFunction(parent) || ts.isFunctionExpression(parent)) && parent.parameters.length === 1
        && ts.isIdentifier(parent.parameters[0].name) && ts.isCallExpression(parent.parent)
        && member(parent.parent.expression, 'then') && parent.parent.arguments.length === 1
        && cleanCallback(parent, builtin) && depends(node, parent.parameters[0].name.text)) {
        const decode = parent.parent.expression.expression;
        if (ts.isCallExpression(decode) && decode.arguments.length === 1
          && member(decode.expression, 'then') && bodyReader(decode.arguments[0])) {
          const request = decode.expression.expression;
          if (ts.isCallExpression(request) && ts.isIdentifier(request.expression)
            && request.expression.text === 'fetch') return true;
        }
      }
      parent = parent.parent;
    }
    return false;
  }

  function credentialSource(node, call) {
    if (ts.isStringLiteralLike(node)) return 'embedded_literal';
    function responseDerived(value) {
      if (ts.isIdentifier(value)) return fromResponse(value, call);
      if (ts.isParenthesizedExpression(value)) return responseDerived(value.expression);
      if (ts.isPropertyAccessExpression(value)) return responseDerived(value.expression);
      if (ts.isElementAccessExpression(value)) {
        return (ts.isStringLiteralLike(value.argumentExpression) || ts.isNumericLiteral(value.argumentExpression))
          && responseDerived(value.expression);
      }
      return false;
    }
    if (responseDerived(node)) return 'earlier_response';
    function simple(value) {
      if (ts.isIdentifier(value) || ts.isStringLiteralLike(value) || ts.isNumericLiteral(value)) return true;
      if (ts.isParenthesizedExpression(value)) return simple(value.expression);
      return ts.isBinaryExpression(value) && value.operatorToken.kind === ts.SyntaxKind.PlusToken
        && simple(value.left) && simple(value.right);
    }
    if (!simple(node)) return null;
    const origins = new Set();
    let unsupported = false;
    walk(node, child => {
      if (ts.isIdentifier(child) && !(ts.isPropertyAccessExpression(child.parent) && child.parent.name === child)) {
        if (fromResponse(child, call)) { origins.add('earlier_response'); return; }
        const value = resolve(child, call.pos);
        if (!value) {
          if (writes.has(child.text) || unsafeWrites.has(child.text)) unsupported = true;
          else origins.add('unavailable_context');
        }
        else if (responseDerived(value)) origins.add('earlier_response');
        else if (ts.isStringLiteralLike(value) || ts.isNumericLiteral(value)) origins.add('local_configuration');
        else unsupported = true;
      }
      if (ts.isCallExpression(child) || ts.isNewExpression(child)) unsupported = true;
    });
    if (unsupported || origins.size > 1) return null;
    return origins.size ? [...origins][0] : 'embedded_literal';
  }

  function bodyValue(node, name) {
    if (ts.isIdentifier(node)) return node.text === name;
    if (ts.isParenthesizedExpression(node)) return bodyValue(node.expression, name);
    if (ts.isPropertyAccessExpression(node) || ts.isElementAccessExpression(node)) return bodyValue(node.expression, name);
    if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.PlusToken) {
      return bodyValue(node.left, name) || bodyValue(node.right, name);
    }
    if (ts.isTemplateExpression(node)) return node.templateSpans.some(span => bodyValue(span.expression, name));
    if (!ts.isCallExpression(node)) return false;
    const fn = node.expression;
    if (ts.isIdentifier(fn) && builtin(fn.text)
      && ['atob', 'btoa', 'encodeURIComponent', 'decodeURIComponent'].includes(fn.text)) {
      return node.arguments.length === 1 && bodyValue(node.arguments[0], name);
    }
    if (ts.isPropertyAccessExpression(fn)) {
      if (fn.expression.getText(file) === 'JSON' && fn.name.text === 'stringify'
        && builtin('JSON')) return node.arguments.length > 0 && bodyValue(node.arguments[0], name);
      return bodyMethods.has(fn.name.text)
        && bodyValue(fn.expression, name);
    }
    return false;
  }

  function responseUse(call) {
    const decodeAccess = call.parent, decodeCall = decodeAccess?.parent;
    if (!decodeAccess || !member(decodeAccess, 'then') || !ts.isCallExpression(decodeCall)) return null;
    const decode = decodeCall.arguments[0];
    if (decodeCall.arguments.length !== 1 || !bodyReader(decode)) return null;
    const useAccess = decodeCall.parent, useCall = useAccess?.parent;
    if (!useAccess || !member(useAccess, 'then') || !ts.isCallExpression(useCall)) return null;
    const callback = useCall.arguments[0];
    if (!callback || !ts.isArrowFunction(callback) || useCall.arguments.length !== 1
      || !cleanCallback(callback, builtin)) return null;
    const name = callback.parameters[0].name.text, sinks = new Set();
    let characterAccess = false, imageWrite = false;
    let unsupported = false;
    walk(callback.body, node => {
      if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.CommaToken) unsupported = true;
      if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.EqualsToken
        && ts.isPropertyAccessExpression(node.left) && ['innerText', 'textContent', 'innerHTML'].includes(node.left.name.text)
        && builtin('document')
        && node.left.expression.getText(file).startsWith('document.') && bodyValue(node.right, name)) sinks.add('page_text');
      if (!ts.isCallExpression(node)) return;
      const fn = node.expression;
      if (ts.isPropertyAccessExpression(fn)) {
        const owner = fn.expression.getText(file);
        if (owner === 'document' && builtin('document')
          && ['write', 'writeln'].includes(fn.name.text) && node.arguments.some(arg => bodyValue(arg, name))) sinks.add('page_text');
        if (owner === 'console' && builtin('console')
          && ['log', 'info', 'error'].includes(fn.name.text) && node.arguments.some(arg => bodyValue(arg, name))) sinks.add('console_output');
        if (fn.name.text === 'charCodeAt' && depends(fn.expression, name)) characterAccess = true;
        if (fn.name.text === 'putImageData') imageWrite = true;
      }
      if (ts.isIdentifier(fn) && fn.text === 'eval' && builtin('eval')
        && node.arguments.some(arg => bodyValue(arg, name))) sinks.add('code_execution');
      if (ts.isIdentifier(fn) && fn.text === 'fetch') {
        const options = object(node.arguments[1], node.pos);
        if (options?.has('body') && bodyValue(options.get('body'), name)) sinks.add('onward_request');
      }
    });
    // Co-occurrence of charCodeAt and putImageData is insufficient to prove
    // their data dependence. Those cases await a dedicated flow analysis.
    if (characterAccess && imageWrite) return null;
    const start = [...text.slice(0, callback.getStart(file))].length;
    const end = [...text.slice(0, callback.end)].length;
    return !unsupported && sinks.size === 1 ? { label: [...sinks][0], span: [start, end] } : null;
  }

  function responseBehaviors(call) {
    // Completeness covers this callback and these five operations only. It is
    // not a claim about execution, hidden helpers, or later promise callbacks.
    const span = node => node ? [[...text.slice(0, node.getStart(file))].length, [...text.slice(0, node.end)].length] : null;
    const result = { version: 1, scope: 'immediate_body_callback', complete: false,
      semantics: 'standard_unmodified_builtins_if_reached', reason: null,
      request_span: span(call), body_reader_span: null, callback_span: null, parameter_span: null, operations: [] };
    const reject = reason => ({ ...result, complete: false, reason, operations: [] });
    const decode = call.parent?.parent;
    if (!member(call.parent, 'then') || !decode || !ts.isCallExpression(decode)
      || decode.arguments.length !== 1 || !bodyReader(decode.arguments[0])) return reject('unsupported_body_reader');
    result.body_reader_span = span(decode.arguments[0]);
    const use = decode.parent?.parent;
    if (!member(decode.parent, 'then') || !use || !ts.isCallExpression(use)
      || use.arguments.length !== 1 || !ts.isArrowFunction(use.arguments[0])) return reject('unsupported_body_callback');
    const callback = use.arguments[0];
    result.callback_span = span(callback);
    result.parameter_span = span(callback.parameters[0]?.name);
    if (!cleanCallback(callback, builtin) || callback.modifiers?.length) return reject('unsupported_control_or_mutation');
    let builtinHazard = [...mutatedBuiltins].some(name => globals.has(name));
    walk(file, node => {
      if (ts.isDeleteExpression(node) && [...changedNames(node.expression)].some(name => globals.has(name))) builtinHazard = true;
      if (!ts.isCallExpression(node)) return;
      if (ts.isIdentifier(node.expression) && node.expression.text === 'eval'
        && (node.pos < callback.pos || node.end > callback.end)) builtinHazard = true;
      // Passing a built-in object to another call may replace its methods.
      if (node.arguments.some(arg => ts.isIdentifier(arg) && (globals.has(arg.text) || arg.text === 'globalThis')
        || ts.isPropertyAccessExpression(arg) && ts.isIdentifier(arg.expression)
          && (globals.has(arg.expression.text) || arg.expression.text === 'globalThis'))) builtinHazard = true;
    });
    if (builtinHazard || !builtin('fetch')) return reject('visible_builtin_hazard');
    const name = callback.parameters[0].name.text;
    const bodyKind = decode.arguments[0].body.expression.name.text === 'text' ? 'text' : 'unknown';
    const forbiddenKeys = new Set(['constructor', 'prototype', '__proto__', 'toString', 'valueOf',
      'toLocaleString', 'hasOwnProperty', 'isPrototypeOf', 'propertyIsEnumerable',
      '__defineGetter__', '__defineSetter__', '__lookupGetter__', '__lookupSetter__']);
    const merge = (parts, kind = 'unknown') => parts.every(Boolean)
      ? { kind, dependencies: parts.flatMap(part => part.dependencies) } : null;
    const literal = node => ts.isStringLiteralLike(node) || ts.isNumericLiteral(node)
      || [ts.SyntaxKind.TrueKeyword, ts.SyntaxKind.FalseKeyword, ts.SyntaxKind.NullKeyword].includes(node.kind);

    function value(node) {
      if (!node) return null;
      if (literal(node)) return { kind: ts.isStringLiteralLike(node) ? 'text' : 'scalar', dependencies: [] };
      if (ts.isIdentifier(node)) return node.text === name ? { kind: bodyKind, dependencies: [span(node)] } : null;
      if (ts.isParenthesizedExpression(node)) return value(node.expression);
      if (ts.isPropertyAccessExpression(node) && !node.questionDotToken && !forbiddenKeys.has(node.name.text)) {
        const base = value(node.expression);
        if (base?.kind === 'text' && node.name.text !== 'length') return null;
        return base?.dependencies.length ? { ...base, kind: node.name.text === 'length' ? 'scalar' : 'unknown' } : null;
      }
      if (ts.isElementAccessExpression(node) && !node.questionDotToken && literal(node.argumentExpression)
        && !forbiddenKeys.has(node.argumentExpression.text)) {
        const base = value(node.expression);
        if (base?.kind === 'text' && !/^\d+$/.test(node.argumentExpression.text)) return null;
        return base?.dependencies.length ? { ...base, kind: 'unknown' } : null;
      }
      if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.PlusToken) {
        const parts = [value(node.left), value(node.right)];
        return merge(parts, parts.some(part => part?.kind === 'text') ? 'text' : 'unknown');
      }
      if (ts.isTemplateExpression(node)) return merge(node.templateSpans.map(part => value(part.expression)), 'text');
      if (!ts.isCallExpression(node) || node.questionDotToken || node.arguments.some(ts.isSpreadElement)) return null;
      const fn = node.expression, args = node.arguments.map(value);
      if (ts.isIdentifier(fn) && builtin(fn.text)
        && ['atob', 'btoa', 'encodeURIComponent', 'decodeURIComponent'].includes(fn.text) && args.length === 1) return merge(args, 'text');
      if (ts.isPropertyAccessExpression(fn) && !fn.questionDotToken) {
        if (fn.expression.getText(file) === 'JSON' && fn.name.text === 'stringify' && builtin('JSON')
          && args.length === 1) return merge(args, 'text');
        const receiver = value(fn.expression);
        const methods = { slice: 'text', substring: 'text', substr: 'text', toString: 'text' };
        // Only native string receivers and literal arguments. Object methods,
        // replacer callbacks, aliases and prototype access are not interpreted.
        const integers = node.arguments.every(arg => ts.isNumericLiteral(arg) && Number.isSafeInteger(Number(arg.text)));
        const nonemptyRange = node.arguments.length < 2 || (fn.name.text === 'substr'
          ? Number(node.arguments[1].text) > 0 : Number(node.arguments[1].text) > Number(node.arguments[0].text));
        if (receiver?.kind === 'text' && methods[fn.name.text] && node.arguments.length <= 2
          && (fn.name.text === 'toString' ? node.arguments.length === 0 : integers && nonemptyRange)) {
          return merge([receiver, ...args], methods[fn.name.text]);
        }
      }
      return null;
    }

    function domReceiver(node) {
      if (!builtin('document')) return false;
      if (ts.isPropertyAccessExpression(node) && !node.questionDotToken
        && ts.isIdentifier(node.expression) && node.expression.text === 'document') {
        return ['body', 'documentElement'].includes(node.name.text);
      }
      return ts.isCallExpression(node) && !node.questionDotToken
        && ts.isPropertyAccessExpression(node.expression) && !node.expression.questionDotToken
        && node.expression.expression.getText(file) === 'document'
        && node.expression.name.text === 'querySelector' && node.arguments.length === 1
        && ts.isStringLiteralLike(node.arguments[0]) && ['body', 'html'].includes(node.arguments[0].text);
    }

    function operation(kind, sink, expression, info) {
      if (!info) return false;
      if (info.dependencies.length) result.operations.push({ operation: kind,
        sink_span: span(sink), value_span: span(expression), dependency_spans: info.dependencies });
      return true;
    }

    function expression(node) {
      if (ts.isParenthesizedExpression(node)) return expression(node.expression);
      if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.EqualsToken
        && ts.isPropertyAccessExpression(node.left) && !node.left.questionDotToken && domReceiver(node.left.expression)) {
        const kind = { innerText: 'plain_text_display', textContent: 'plain_text_display', innerHTML: 'html_parse' }[node.left.name.text];
        return Boolean(kind) && operation(kind, node, node.right, value(node.right));
      }
      if (!ts.isCallExpression(node)) return Boolean(value(node));
      if (node.questionDotToken || node.arguments.some(ts.isSpreadElement)) return false;
      const fn = node.expression;
      if (ts.isPropertyAccessExpression(fn) && !fn.questionDotToken) {
        const owner = fn.expression.getText(file);
        const kind = owner === 'document' && builtin(owner) && ['write', 'writeln'].includes(fn.name.text) ? 'html_parse'
          : owner === 'console' && builtin(owner) && ['log', 'info', 'error'].includes(fn.name.text) ? 'console_output' : null;
        if (kind) return node.arguments.every(arg => operation(kind, node, arg, value(arg)));
      }
      if (ts.isIdentifier(fn) && fn.text === 'eval' && builtin('eval')) {
        const info = node.arguments.length === 1 ? value(node.arguments[0]) : null;
        // Constant eval strings may reference the callback parameter by name.
        // Never turn those, or non-string eval arguments, into negative labels.
        return Boolean(info?.dependencies.length && info.kind === 'text')
          && operation('code_execution', node, node.arguments[0], info);
      }
      if (ts.isIdentifier(fn) && fn.text === 'fetch' && builtin('fetch')) {
        const options = node.arguments.length === 1 ? new Map() : object(node.arguments[1], node.pos);
        if (node.arguments.length < 1 || node.arguments.length > 2 || !options
          || !ts.isStringLiteralLike(node.arguments[0])) return false;
        for (const [key, item] of options) {
          if (key === 'body') {
            if (!operation('onward_request_body', node, item, value(item))) return false;
          } else if (key === 'headers') {
            const headers = object(item, node.pos);
            if (!headers || ![...headers.values()].every(header => value(header))) return false;
          } else if (!literal(item)) return false;
        }
        return true;
      }
      return Boolean(value(node));
    }

    const statements = ts.isBlock(callback.body) ? callback.body.statements : null;
    if (statements) {
      for (const statement of statements) {
        if (ts.isEmptyStatement(statement)) continue;
        if (!(ts.isExpressionStatement(statement) || ts.isReturnStatement(statement))
          || !statement.expression || !expression(statement.expression)) return reject('unsupported_statement_or_value');
      }
    } else if (!expression(callback.body)) return reject('unsupported_statement_or_value');
    return { ...result, complete: true };
  }

  const contracts = calls.map((call, index) => {
    const url = resolve(call.arguments[0], call.pos);
    const knownUrl = url && ts.isStringLiteralLike(url);
    const options = call.arguments.length < 2 ? new Map() : object(call.arguments[1], call.pos);
    const methodNode = options?.get('method');
    const resolvedMethod = methodNode ? resolve(methodNode, call.pos) : null;
    const method = options && !methodNode && knownUrl ? 'GET' : resolvedMethod && ts.isStringLiteralLike(resolvedMethod) ? resolvedMethod.text.toUpperCase() : null;
    const headers = options && !options.has('headers') ? new Map() : options ? object(options.get('headers'), call.pos) : null;
    const credentials = headers ? [...headers].filter(([key]) => credentialHeaders.has(key.toLowerCase())) : null;
    const source = credentials?.length === 0 && knownUrl ? 'no_configured_credential_header'
      : credentials?.length === 1 ? credentialSource(credentials[0][1], call) : null;
    const span = node => [[...text.slice(0, node.getStart(file))].length, [...text.slice(0, node.end)].length];
    return { ordinal: index + 1, span: span(call), method,
      credential_source: source, credential_header: credentials?.length === 1 ? credentials[0][0] : null,
      credential_span: credentials?.length === 1 ? span(credentials[0][1]) : null,
      response_use: responseUse(call),
      response_behaviors: responseBehaviors(call),
      focal_expression: call.expression.getText(file) };
  });
  return { admitted: true, language: 'javascript', template, calls: contracts };
}

for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  const row = JSON.parse(line);
  const result = analyze(row.text);
  if (process.argv.includes('--fingerprints-only')) {
    process.stdout.write(JSON.stringify({ id: row.id, text_sha256: hash(row.text), template: result.template ?? null }) + '\n');
    continue;
  }
  process.stdout.write(JSON.stringify({ id: row.id, text_sha256: hash(row.text), ...result }) + '\n');
}
