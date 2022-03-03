/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @format
 */

const React = require('react');

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">Tutorials</h1>
            </header>
            <p>
              This is the tutorials page. Navigate the sidebar to find various
              tutorials.
            </p>
            <h4>External Blog Posts</h4>
            <p>
              <a
                href="https://ai.facebook.com/blog/introducing-opacus-a-high-speed-library-for-training-pytorch-models-with-differential-privacy/"
                target="_blank">
                Introducing Opacus
              </a>
              , by Meta AI
            </p>
            <h4>Differential Privacy Blog Post Series</h4>
            <ol>
              <li>
                <a
                  href="https://bit.ly/dp-sgd-algorithm-explained"
                  target="_blank">
                  DP-SGD Algorithm Explained
                </a>
              </li>
              <li>
                <a
                  href="https://bit.ly/per-sample-gradient-computing-opacus"
                  target="_blank">
                  Efficient Per-Sample Gradient Computation in Opacus
                </a>
              </li>
            </ol>
            <h4>Videos*</h4>
            <p>* Note that Opacus API has changed over time and some of the code samples and demos in the videos may not work. The concepts presented in the videos though are concrete and still valid.</p>
            <ol>
              <li>
                <a
                  href="https://www.youtube.com/watch?v=MWPwofiQMdE&list=PLUNOsx6Az_ZGKQd_p4StdZRFQkCBwnaY6&index=51"
                  target="_blank">
                  OpenMined PriCon 2020 Tutorial: DP Model Training with Opacus
                </a>
              </li>
              <li>
                <a
                  href="https://www.youtube.com/watch?v=l6fbl2CBnq0"
                  target="_blank">
                  PyTorch Developer Day 2020: Differential Privacy on PyTorch
                </a>
              </li>
            </ol>
            <h4>Blog Posts by OpenMined</h4>
            <ol>
              <li>
                <a
                  href="https://blog.openmined.org/differentially-private-deep-learning-using-opacus-in-20-lines-of-code/"
                  target="_blank">
                  Differentially Private Deep Learning In 20 Lines Of Code
                </a>
              </li>
              <li>
                <a
                  href="https://blog.openmined.org/pysyft-opacus-federated-learning-with-differential-privacy/"
                  target="_blank">
                  PySyft + Opacus: Federated Learning With Differential Privacy
                </a>
              </li>
            </ol>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
