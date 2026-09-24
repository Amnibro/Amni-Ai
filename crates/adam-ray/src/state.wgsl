fn upd(s: ptr<function, array<u32, 40>>, bb: u32) {
    var fx = bitcast<f32>((*s)[37]);
    var fy = bitcast<f32>((*s)[38]);
    var wh = (*s)[16]; var pw1 = (*s)[17]; var pw2 = (*s)[18]; var pw3 = (*s)[19]; var pw4 = (*s)[20]; var s1 = (*s)[21]; var s2 = (*s)[22]; var g1 = (*s)[23]; var at = (*s)[24]; var ind = (*s)[25]; var g2 = (*s)[26]; var col = (*s)[27]; var inc = (*s)[28];
    var lastp = (*s)[29] & 0xFFFFu; var tick = (*s)[29] >> 16u;
    var inq = (*s)[30]; var q1 = (*s)[31]; var q2 = (*s)[32]; var c1 = (*s)[33]; var c2 = (*s)[34]; var qon = (*s)[35]; var wlen = (*s)[36];
    let isw = bt(WORD, bb);
    if (!isw && wh != 0u) {
        pw4 = pw3; pw3 = pw2; pw2 = pw1; pw1 = wh; s2 = s1; g2 = g1;
        let lx = textureLoad(lex, vec2<u32>((pw1 & WM) % 1024u, (pw1 & WM) / 1024u), 0);
        let tr = textureLoad(trg, vec2<u32>(pw1 % 360u, 0u), 0);
        s1 = lx.x; g1 = lx.y; fx = TA * fx + tr.x * TC * 8.0; fy = TA * fy + tr.y * TC * 8.0;
        if (QT && inq == 1u && wlen >= 3u && !(bb == 58u && (*s)[0] == 109u && (*s)[1] == 97u && (*s)[2] == 100u && (*s)[3] == 65u) && lx.z > 0u) {
            if (c1 == 0u || lx.z < c1) { q2 = q1; c2 = c1; q1 = pw1; c1 = lx.z; } else if (pw1 != q1 && (c2 == 0u || lx.z < c2)) { q2 = pw1; c2 = lx.z; }
        }
    }
    wlen = select(wlen, select(0u, wlen + 1u, isw), QT);
    wh = select(0u, wh * 83492791u + bb + 1u, isw);
    let nl = bb == 10u;
    col = select(col + 1u, 0u, nl);
    if (nl) { at = 1u; ind = 0u; } else if (bb == 32u) { ind += select(0u, 1u, at == 1u); } else { at = 0u; }
    tick = select(select(tick, 0u, nl), tick + 1u, bb == 96u);
    if (tick == 3u) { inc = 1u - inc; tick = 0u; }
    lastp = select(select(lastp, 10u, nl), bb, bt(PUNC, bb));
    (*s)[15] = (*s)[14]; (*s)[14] = (*s)[13]; (*s)[13] = (*s)[12]; (*s)[12] = (*s)[11]; (*s)[11] = (*s)[10]; (*s)[10] = (*s)[9]; (*s)[9] = (*s)[8]; (*s)[8] = (*s)[7]; (*s)[7] = (*s)[6]; (*s)[6] = (*s)[5]; (*s)[5] = (*s)[4]; (*s)[4] = (*s)[3]; (*s)[3] = (*s)[2]; (*s)[2] = (*s)[1]; (*s)[1] = (*s)[0];
    (*s)[0] = bb;
    if (QT) {
        if ((*s)[5] == 85u && (*s)[4] == 115u && (*s)[3] == 101u && (*s)[2] == 114u && (*s)[1] == 58u && (*s)[0] == 32u) { q1 = 0u; q2 = 0u; c1 = 0u; c2 = 0u; inq = 1u; qon = 0u; }
        if ((*s)[5] == 65u && (*s)[4] == 100u && (*s)[3] == 97u && (*s)[2] == 109u && (*s)[1] == 58u && (*s)[0] == 32u) { inq = 0u; qon = select(0u, 1u, q1 != 0u); }
        if ((*s)[0] == 10u && (*s)[1] == 10u) { qon = 0u; }
    }
    (*s)[16] = wh; (*s)[17] = pw1; (*s)[18] = pw2; (*s)[19] = pw3; (*s)[20] = pw4; (*s)[21] = s1; (*s)[22] = s2; (*s)[23] = g1; (*s)[24] = at; (*s)[25] = ind; (*s)[26] = g2; (*s)[27] = col; (*s)[28] = inc; (*s)[29] = lastp | (tick << 16u);
    (*s)[30] = inq; (*s)[31] = q1; (*s)[32] = q2; (*s)[33] = c1; (*s)[34] = c2; (*s)[35] = qon; (*s)[36] = wlen; (*s)[37] = bitcast<u32>(fx); (*s)[38] = bitcast<u32>(fy);
}
fn plan(s: ptr<function, array<u32, 40>>) {
    let wh = (*s)[16]; let pw1 = (*s)[17]; let pw2 = (*s)[18]; let pw3 = (*s)[19]; let pw4 = (*s)[20];
    let h5 = hm(hm((*s)[4], (*s)[3], 17u), hm((*s)[2], (*s)[1], 19u), 21u);
    let h6 = hm(h5, (*s)[5], 23u);
    let h8 = hm(hm(h6, (*s)[6], 29u), (*s)[7], 31u);
    let h10 = hm(hm(h8, (*s)[8], 41u), (*s)[9], 43u);
    let h12 = hm(hm(h10, (*s)[10], 47u), (*s)[11], 53u);
    let h16 = hm(hm(hm(hm(h12, (*s)[12], 59u), (*s)[13], 61u), (*s)[14], 67u), (*s)[15], 71u);
    var ku = array<vec2<u32>, 18>(vec2<u32>(hm((*s)[2], (*s)[1], 7u), (*s)[0]), vec2<u32>(hm(hm((*s)[3], (*s)[2], 9u), (*s)[1], 11u), (*s)[0]), vec2<u32>(wh, pw1 & 255u), vec2<u32>(pw1 & 4095u, pw2 & 4095u), vec2<u32>(hm(min((*s)[25], 32u), (*s)[28] * 512u + ((*s)[29] & 0xFFFFu), 13u), min((*s)[27], 255u)), vec2<u32>(h5, (*s)[0]), vec2<u32>(h6, (*s)[0]), vec2<u32>(h8, (*s)[0]), vec2<u32>(hm(pw1, pw2, 37u) & 65535u, pw3 & 4095u), vec2<u32>(h10, (*s)[0]), vec2<u32>(h12, (*s)[0]), vec2<u32>(h16, (*s)[0]), vec2<u32>(hm(hm(pw1, pw2, 73u), hm(pw3, pw4, 79u), 83u), wh & 65535u), vec2<u32>(wh, 0u), vec2<u32>(hm(wh, pw1, 89u), 0u), vec2<u32>((*s)[21], (*s)[22]), vec2<u32>((*s)[23], (*s)[26]), vec2<u32>(wh, (*s)[21]));
    for (var l = 2u; l < BASE - 1u; l++) { put(l, 0u, select(0u, sl(ku[l - 2u].x, ku[l - 2u].y, l), has(l)), 1.0); }
    let fx = bitcast<f32>((*s)[37]);
    let fy = bitcast<f32>((*s)[38]);
    let a0 = at2(fy, fx) / TAU;
    let uu = (a0 - floor(a0)) * 256.0;
    let vv = min(sqrt(fx * fx + fy * fy), 7.99) / 8.0 * 256.0;
    let u0 = floor(uu);
    let v0 = floor(vv);
    let fu = uu - u0;
    let fv = vv - v0;
    for (var k = 0; k < 4; k++) {
        let w = select(1.0 - fu, fu, (k & 1) == 1) * select(1.0 - fv, fv, (k >> 1u) == 1);
        if (w != 0.0) { put(BASE - 1u, u32(k), sl(u32((i32(u0) + (k & 1)) % 256), u32(min(i32(v0) + (k >> 1u), 255)), BASE - 1u), w); }
    }
    if (QT && (*s)[35] == 1u) { let q1 = (*s)[31]; put(BASE, 0u, sl(q1, wh, BASE), 1.0); put(BASE + 1u, 0u, sl(hm(q1, (*s)[32], 107u), pw1 & 4095u, BASE + 1u), 1.0); put(BASE + 2u, 0u, sl(q1, (*s)[0] + ((*s)[1] << 8u), BASE + 2u), 1.0); }
}
