//! Simulation
//! Manages the evolution of population over multiple generations

use std::io;
use std::time::Instant;

use crate::pop::Population;
use crate::viz::Viz;

pub struct Simulation {
    // 生成次数
    gen_count: usize,

    pop: Population,
    // UI
    viz: Viz,
    // 生成开始的时刻
    gen_start_ts: Instant,
    //最大积分
    max_score: usize,
}

#[derive(Default, Clone, Copy)]
pub struct GenerationSummary {
    // 生成次数
    pub gen_count: usize,
    // 耗时
    pub time_elapsed_secs: f32,
    // 本次最大积分
    pub gen_max_score: usize,
    // 总最大积分
    pub sim_max_score: usize,
}

impl Simulation {
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            gen_count: 0,
            pop: Population::new(),
            viz: Viz::new()?,
            gen_start_ts: Instant::now(),
            max_score: 0,
        })
    }

    pub fn stop(&self) -> io::Result<()> {
        Viz::restore_terminal()
    }

    pub fn update(&mut self) {
        let games_alive = self.pop.update();
        if games_alive == 0 {
            self.end_current_genration();
            self.start_new_generation();
        }

        self.viz.update();
    }

    pub fn draw(&mut self) {
        self.viz.draw();
    }

    pub fn start_new_generation(&mut self) {
        self.gen_count += 1;
        self.pop.reset();
    }

    pub fn end_current_genration(&mut self) {
        let (best_net, gen_max_score) = self.pop.get_gen_summary();
        if gen_max_score > self.max_score {
            self.max_score = gen_max_score;
            best_net.save();
            self.viz.update_brain(best_net);
        }

        let stats = GenerationSummary {
            gen_count: self.gen_count,
            time_elapsed_secs: self.gen_start_ts.elapsed().as_secs_f32(),
            gen_max_score,
            sim_max_score: self.max_score,
        };
        self.viz
            .update_summary(stats, self.pop.mutation_rate, self.pop.mutation_magnitude);
        self.gen_start_ts = Instant::now();
    }
}
