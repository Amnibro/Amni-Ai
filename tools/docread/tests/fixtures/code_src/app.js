const fs = require("fs");

function load(path) {
  return JSON.parse(fs.readFileSync(path, "utf8"));
}

const area = function (p) {
  return p.w * p.h;
};

const double = (x) => x * 2;

module.exports = { load, area, double };
