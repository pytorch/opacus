/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
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
            <h2>External Blog Posts</h2>
            <p>
              <a
                href="https://ai.facebook.com/blog/introducing-opacus-a-high-speed-library-for-training-pytorch-models-with-differential-privacy/"
                target="_blank">
                Introducing Opacus
              </a>
              , by Facebook AI
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
