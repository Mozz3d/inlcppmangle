import {exit} from "process";
import {join} from "path";
import {readFileSync, readdirSync, writeFileSync} from "fs";

const rootNDB = "<path>\\cp2077-nativedb\\src\\assets\\reddump";
const rootSDK = "<path>\\RED4ext_SDK";

if (rootNDB.includes("<path>")) {
  console.error("[error] You must define the path to JSON files of NativeDB.");
  exit(1);
}
if (rootSDK.includes("<path>")) {
  console.error("[error] You must define the path to RED4ext.SDK.");
  exit(1);
}

const namespaces = getNamespaces(join(rootSDK, "include", "RED4ext", "Scripting", "Natives", "Generated"));
const prefixNamespaces = namespaces.map((ns) => ns.replaceAll("::", "")).sort((a, b) => b.length - a.length);

const words = {};
getEnums(words, readDump(join(rootNDB, "enums.json")));
getBitfields(words, readDump(join(rootNDB, "bitfields.json")));
getGlobals(words, readDump(join(rootNDB, "globals.json")));
getClasses(words, readDump(join(rootNDB, "classes.json")));

const dictionary = Object.keys(words).sort();
writeFileSync("./cp2077-dictionary-namespaces.txt", namespaces.join("\n"), {encoding: "utf8"});
writeFileSync("./cp2077-dictionary-ndb.txt", dictionary.join("\n"), {encoding: "utf8"});

console.log(`Found ${namespaces.length} namespaces`);
console.log(`Found ${dictionary.length} words`);

function readDump(path) {
  const data = readFileSync(path, {encoding: "utf8"});
  return JSON.parse(data);
}

function getNamespaces(path, parent) {
  parent ??= "";

  const namespaces = [];
  const dirents = readdirSync(path, {withFileTypes: true});

  for (const dirent of dirents) {
    if (dirent.isDirectory()) {
      let name = dirent.name;
      if (parent.length > 0) {
        name = `${parent}::${name}`;
      }
      namespaces.push(name);
      namespaces.push(...getNamespaces(join(path, dirent.name), name));
    }
  }

  return namespaces.filter((ns) => ns !== "$");
}

function addWords(words, tokens) {
  if (!tokens) {
    return;
  }

  for (const token of tokens) {
    if (!(token in tokens)) {
      words[token] = 0;
    }
  }
}

function sanitizeWords(word) {
  if (word === undefined) {
    return undefined;
  }

  if (!Number.isNaN(parseInt(word))) {
    return undefined;
  }

  return word.split(";")
             .flatMap((word) => word.split("."))
             .flatMap((word) => word.split(">"))
             .flatMap((word) => word.split("<"))
             .flatMap((word) => word.split("_"))
             .flatMap((word) => word.split(" "))
             .flatMap((word) => word.split("::"))
             .flatMap((word) => word.split(">>"))
             .flatMap((word) => word.split(/(?=[A-Z])/))
             .map((word) => word.trim())
             .map(sanitizeWord)
             .filter((word) => word && word.length > 1);
}

function sanitizeWord(word) {
  if (word === undefined) {
    return undefined;
  }

  if (!Number.isNaN(parseInt(word))) {
    return undefined;
  }

  const ns = prefixNamespaces.find((ns) => word.startsWith(ns));
  if (ns) {
    word = word.substring(ns.length);
  }
  return word;
}

function getEnums(words, nodes) {
  for (const node of nodes) {
    const a = sanitizeWords(node.a);
    const b = sanitizeWords(node.b);
    addWords(words, a);
    addWords(words, b);
  
    for (let member of Object.keys(node.c)) {
      member = sanitizeWords(member);
      addWords(words, member);
    }
  }
}

function getBitfields(words, nodes) {
  return getEnums(words, nodes);
}

function getGlobals(words, nodes) {
  for (const node of nodes) {
    getFunction(words, node);
  }
}

function getFunction(words, node) {
  let name = sanitizeWords(node.a);
  addWords(words, name);

  node.e ??= [];
  for (const arg of node.e) {
    name = sanitizeWords(arg.b);
    addWords(words, name);
  }
}

function getClasses(words, nodes) {
  for (const node of nodes) {
    getClass(words, node);
  }
}

function getClass(words, node) {
  let name = sanitizeWords(node.b);
  addWords(words, name);

  node.e ??= [];
  for (const prop of node.e) {
    name = sanitizeWords(prop.b);
    addWords(words, name);
  }

  node.f ??= [];
  for (const fn of node.f) {
    getFunction(words, fn);
  }
}
