<?php

namespace Rubix\ML\Benchmarks\Classifiers;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Datasets\Generators\Agglomerate;

/**
 * @Groups({"Classifiers"})
 */
class LogisticRegressionBench
{
    protected const TRAINING_SIZE = 2500;

    protected const TESTING_SIZE = 10000;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    public $training;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    public $testing;

    /**
     * @var \Rubix\ML\Classifiers\LogisticRegression
     */
    protected $estimator;

    public function setUpTrainPredict() : void
    {
        $generator = new Agglomerate([
            'Iris-setosa' => new Blob([5.0, 3.42, 1.46, 0.24], [0.35, 0.38, 0.17, 0.1]),
            'Iris-virginica' => new Blob([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
        ]);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new LogisticRegression();
    }

    /**
     * @Iterations(3)
     * @BeforeMethods({"setUpTrainPredict"})
     * @OutputTimeUnit("seconds", precision=3)
     */
    public function bench_train_predict() : void
    {
        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}
