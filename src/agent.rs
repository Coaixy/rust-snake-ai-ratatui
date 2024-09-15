//! AI Agent
//! An instance of the Game played by an AI

use nn::Net;

use crate::game::Game;
use crate::{get_eight_dirs, nn, FourDirs, Point, NN_ARCH, NUM_STEPS};


// 个体对象
#[derive(Clone)]
pub struct Agent {
    // 游戏实例
    pub game: Game,
    // 神经网络
    pub brain: Net,
}

impl Agent {
    #[must_use]
    pub fn new(is_load: bool) -> Self {
        /***
            如果启动了读取的话就加载神经网络
            否则的话生成新的网络
         */
        let brain = if is_load {
            let mut net = Net::load();
            net.mutate(0.0, 0.1);
            net
        } else {
            Net::new(&NN_ARCH)
        };

        Self {
            game: Game::new(),
            brain,
        }
    }

    #[must_use]
    pub fn with_brain(brain: Net) -> Self {
        Self {
            game: Game::new(),
            brain,
        }
    }

    pub fn update(&mut self) -> bool {
        if self.game.is_dead {
            return false;
        }

        self.game.update(self.get_brain_output());

        // Limit the number of steps the snake can take without eating
        let step_limit = self.get_step_limit();
        if self.game.no_food_steps >= step_limit {
            self.game.is_dead = true;
        }

        true
    }
    // 适应度函数
    #[must_use]
    pub fn fitness(&self) -> f32 {
        let score = self.game.body.len() as f32;
        if score <= 1.0 {
            return 1.0;
        }

        let mut fitness = 1.0;
        if score < 5.0 {
            fitness *= 2.0_f32.powf(score);
            fitness *= score;
            fitness *= self.game.total_steps as f32 * 0.1;
        } else {
            fitness *= score * score * score;
            fitness *= self.game.total_steps as f32 * 0.1;
        }

        fitness
    }

    #[must_use]
    pub fn get_brain_output(&self) -> FourDirs {
        // 获取输入
        let vision = self.get_brain_input();
        // 预测
        let nn_out = self.brain.predict(vision);
        // 获取最大的方向
        let (l, r, b, t) = (nn_out[0], nn_out[1], nn_out[2], nn_out[3]);
        let mut directions = [
            (l, FourDirs::Left),
            (r, FourDirs::Right),
            (b, FourDirs::Bottom),
            (t, FourDirs::Top),
        ];
        directions.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        directions[0].1
    }

    #[must_use]
    pub fn get_brain_input(&self) -> Vec<f64> {
        // 获取八个方向
        let dirs = get_eight_dirs().to_vec();
        // 获取视野的数据
        let vision = self.get_snake_vision(dirs);
        // 获取头的方向
        let head_dir = self.game.dir.get_one_hot_dir();
        // 获取尾巴的方向
        let tail_dir = self.get_tail_direction().get_one_hot_dir();
        // 合并数据
        vision.into_iter().chain(head_dir).chain(tail_dir).collect()
    }

    fn get_snake_vision(&self, dirs: Vec<(i32, i32)>) -> Vec<f64> {
        let mut vision = Vec::new();

        for d in dirs {
            // Food and Body are one hot
            let (solid, food) = self.vision_in_dir(self.game.head, d);
            vision.push(f64::from(solid));
            vision.push(if food { 1.0 } else { 0.0 });
        }

        vision
    }

    fn vision_in_dir(&self, st: Point, dir: (i32, i32)) -> (f32, bool) {
        let mut food = false;
        // 头的坐标
        let mut temp_pt: Point = st;
        let mut dist = 0;

        loop {
            // 墙或者身体
            if self.game.is_wall(temp_pt) || self.game.is_snake_body(temp_pt) {
                break;
            }
            // 食物
            if self.game.food == temp_pt {
                food = true;
            }
            // 新的坐标 计算出8个未知的坐标
            temp_pt = Point::new(temp_pt.x + dir.0, temp_pt.y + dir.1);

            dist += 1;
            if dist > 1000 {
                break;
            }
        }
        // 返回距离和是否有食物
        (1.0 / dist as f32, food)
    }

    #[must_use]
    pub fn get_step_limit(&self) -> usize {
        match self.game.score() {
            score if score > 30 => NUM_STEPS * 6,
            score if score > 20 => NUM_STEPS * 3,
            score if score > 5 => NUM_STEPS * 2,
            _ => NUM_STEPS,
        }
    }

    fn get_tail_direction(&self) -> FourDirs {
        if let Some(tail) = self.game.body.last() {
            if let Some(body) = self.game.body.get(self.game.body.len() - 2) {
                let x = body.x - tail.x;
                let y = body.y - tail.y;

                return match (x, y) {
                    (-1, 0) => FourDirs::Left,
                    (1, 0) => FourDirs::Right,
                    (0, 1) => FourDirs::Bottom,
                    _ => FourDirs::Top,
                };
            }
        }

        self.game.dir
    }
}

impl PartialEq for Agent {
    fn eq(&self, other: &Self) -> bool {
        self.fitness() == other.fitness()
    }
}

impl PartialOrd for Agent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fitness().partial_cmp(&other.fitness())
    }
}
