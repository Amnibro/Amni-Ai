use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, KeyboardEvent};
use std::cell::RefCell;
use std::rc::Rc;
use std::f64::consts::PI;
const W: f64 = 800.0;
const H: f64 = 600.0;
const SHIP_THRUST: f64 = 0.15;
const SHIP_DRAG: f64 = 0.99;
const SHIP_TURN: f64 = 0.08;
const BULLET_SPEED: f64 = 8.0;
const BULLET_LIFE: u32 = 60;
#[derive(Clone, Copy)]
struct V { x: f64, y: f64 }
impl V {
    fn new(x: f64, y: f64) -> Self { Self { x, y } }
    fn add(&self, o: V) -> V { V::new(self.x + o.x, self.y + o.y) }
    fn wrap(&self) -> V { V::new(self.x.rem_euclid(W), self.y.rem_euclid(H)) }
    fn dist(&self, o: V) -> f64 { ((self.x-o.x).powi(2) + (self.y-o.y).powi(2)).sqrt() }
}
struct Ship { pos: V, vel: V, angle: f64, alive: bool }
struct Asteroid { pos: V, vel: V, size: f64 }
struct Bullet { pos: V, vel: V, life: u32 }
struct Game {
    ship: Ship,
    asteroids: Vec<Asteroid>,
    bullets: Vec<Bullet>,
    score: u32,
    keys: [bool; 256],
    fire_cooldown: u32,
}
impl Game {
    fn new() -> Self {
        let mut g = Self {
            ship: Ship { pos: V::new(W/2.0, H/2.0), vel: V::new(0.0,0.0), angle: 0.0, alive: true },
            asteroids: Vec::new(),
            bullets: Vec::new(),
            score: 0,
            keys: [false; 256],
            fire_cooldown: 0,
        };
        for i in 0..5 { g.spawn_asteroid(40.0, i as f64); }
        g
    }
    fn spawn_asteroid(&mut self, size: f64, seed: f64) {
        let angle = (seed * 1.234567).sin() * PI * 2.0;
        let speed = 1.0 + (seed * 7.13).cos().abs() * 1.5;
        self.asteroids.push(Asteroid {
            pos: V::new((seed * 137.0) % W, (seed * 211.0) % H),
            vel: V::new(angle.cos() * speed, angle.sin() * speed),
            size,
        });
    }
    fn step(&mut self) {
        if self.fire_cooldown > 0 { self.fire_cooldown -= 1; }
        if self.ship.alive {
            if self.keys[37] { self.ship.angle -= SHIP_TURN; }
            if self.keys[39] { self.ship.angle += SHIP_TURN; }
            if self.keys[38] {
                self.ship.vel.x += self.ship.angle.cos() * SHIP_THRUST;
                self.ship.vel.y += self.ship.angle.sin() * SHIP_THRUST;
            }
            if self.keys[32] && self.fire_cooldown == 0 {
                self.bullets.push(Bullet {
                    pos: self.ship.pos,
                    vel: V::new(self.ship.angle.cos() * BULLET_SPEED, self.ship.angle.sin() * BULLET_SPEED),
                    life: BULLET_LIFE,
                });
                self.fire_cooldown = 8;
            }
            self.ship.vel.x *= SHIP_DRAG;
            self.ship.vel.y *= SHIP_DRAG;
            self.ship.pos = self.ship.pos.add(self.ship.vel).wrap();
        }
        for b in &mut self.bullets {
            b.pos = b.pos.add(b.vel).wrap();
            if b.life > 0 { b.life -= 1; }
        }
        self.bullets.retain(|b| b.life > 0);
        for a in &mut self.asteroids {
            a.pos = a.pos.add(a.vel).wrap();
        }
        let mut hit_bullets: Vec<usize> = Vec::new();
        let mut split: Vec<(V, V, f64)> = Vec::new();
        let mut remove_a: Vec<usize> = Vec::new();
        for (i, a) in self.asteroids.iter().enumerate() {
            for (j, b) in self.bullets.iter().enumerate() {
                if a.pos.dist(b.pos) < a.size {
                    hit_bullets.push(j);
                    remove_a.push(i);
                    self.score += (60.0 / a.size) as u32;
                    if a.size > 20.0 {
                        for k in 0..2 {
                            let na = a.vel.y.atan2(a.vel.x) + (k as f64 - 0.5) * 1.2;
                            split.push((a.pos, V::new(na.cos()*2.0, na.sin()*2.0), a.size * 0.6));
                        }
                    }
                    break;
                }
            }
            if self.ship.alive && a.pos.dist(self.ship.pos) < a.size + 10.0 {
                self.ship.alive = false;
            }
        }
        remove_a.sort_unstable(); remove_a.dedup();
        for i in remove_a.iter().rev() { self.asteroids.remove(*i); }
        hit_bullets.sort_unstable(); hit_bullets.dedup();
        for i in hit_bullets.iter().rev() {
            if *i < self.bullets.len() { self.bullets.remove(*i); }
        }
        for (p, v, s) in split { self.asteroids.push(Asteroid { pos: p, vel: v, size: s }); }
        if self.asteroids.is_empty() {
            for i in 0..6 { self.spawn_asteroid(45.0, (self.score as f64) + i as f64); }
        }
    }
    fn draw(&self, ctx: &CanvasRenderingContext2d) {
        ctx.set_fill_style_str("#000");
        ctx.fill_rect(0.0, 0.0, W, H);
        ctx.set_stroke_style_str("#0f0");
        ctx.set_line_width(2.0);
        if self.ship.alive {
            ctx.save();
            let _ = ctx.translate(self.ship.pos.x, self.ship.pos.y);
            let _ = ctx.rotate(self.ship.angle);
            ctx.begin_path();
            ctx.move_to(10.0, 0.0);
            ctx.line_to(-8.0, 6.0);
            ctx.line_to(-5.0, 0.0);
            ctx.line_to(-8.0, -6.0);
            ctx.close_path();
            ctx.stroke();
            ctx.restore();
        }
        ctx.set_stroke_style_str("#fff");
        for a in &self.asteroids {
            ctx.begin_path();
            let _ = ctx.arc(a.pos.x, a.pos.y, a.size, 0.0, PI * 2.0);
            ctx.stroke();
        }
        ctx.set_fill_style_str("#ff0");
        for b in &self.bullets {
            ctx.begin_path();
            let _ = ctx.arc(b.pos.x, b.pos.y, 2.0, 0.0, PI * 2.0);
            ctx.fill();
        }
        ctx.set_fill_style_str("#fff");
        ctx.set_font("16px monospace");
        let _ = ctx.fill_text(&format!("SCORE {}", self.score), 10.0, 20.0);
        if !self.ship.alive {
            ctx.set_font("48px monospace");
            let _ = ctx.fill_text("GAME OVER", W/2.0 - 130.0, H/2.0);
        }
    }
}
#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas = document.get_element_by_id("canvas").ok_or("no #canvas")?;
    let canvas: HtmlCanvasElement = canvas.dyn_into()?;
    canvas.set_width(W as u32);
    canvas.set_height(H as u32);
    let ctx = canvas.get_context("2d")?.ok_or("no ctx")?.dyn_into::<CanvasRenderingContext2d>()?;
    let game = Rc::new(RefCell::new(Game::new()));
    {
        let game = game.clone();
        let cb = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
            let kc = e.key_code() as usize;
            if kc < 256 { game.borrow_mut().keys[kc] = true; }
        });
        document.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let game = game.clone();
        let cb = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
            let kc = e.key_code() as usize;
            if kc < 256 { game.borrow_mut().keys[kc] = false; }
        });
        document.add_event_listener_with_callback("keyup", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    let f = Rc::new(RefCell::new(None as Option<Closure<dyn FnMut()>>));
    let g = f.clone();
    let game_loop = game.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut gm = game_loop.borrow_mut();
            gm.step();
            gm.draw(&ctx);
        }
        let _ = window.request_animation_frame(
            f.borrow().as_ref().unwrap().as_ref().unchecked_ref()
        );
    }));
    let window2 = web_sys::window().unwrap();
    let _ = window2.request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref());
    Ok(())
}
