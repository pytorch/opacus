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
const fs = require('fs-extra');
const path = require('path');
const join = path.join;
const CWD = process.cwd();

const CompLibrary = require(join(
  CWD,
  '/node_modules/docusaurus/lib/core/CompLibrary.js',
));
const SideNav = require(join(
  CWD,
  '/node_modules/docusaurus/lib/core/nav/SideNav.js',
));

const Container = CompLibrary.Container;

const OVERVIEW_ID = 'tutorial_overview';

class TutorialSidebar extends React.Component {
  render() {
    const {currentTutorialID} = this.props;
    const current = {
      id: currentTutorialID || OVERVIEW_ID,
    };

    const toc = [
      {
        type: 'CATEGORY',
        title: 'Tutorials',
        children: [
          {
            type: 'LINK',
            item: {
              permalink: 'tutorials/',
              id: OVERVIEW_ID,
              title: 'Overview',
            },
          },
        ],
      },
    ];

    const jsonFile = join(CWD, 'tutorials.json');
    const normJsonFile = path.normalize(jsonFile);
    const json = JSON.parse(fs.readFileSync(normJsonFile, {encoding: 'utf8'}));

    Object.keys(json).forEach(category => {
      const categoryItems = json[category];
      const items = [];
      categoryItems.map(item => {
        items.push({
          type: 'LINK',
          item: {
            permalink: `tutorials/${item.id}`,
            id: item.id,
            title: item.title,
          },
        });
      });

      toc.push({
        type: 'CATEGORY',
        title: category,
        children: items,
      });
    });

    return (
      <Container className="docsNavContainer" id="docsNav" wrapper={false}>
        <SideNav
          language={'tutorials'}
          root={'tutorials'}
          title="Tutorials"
          contents={toc}
          current={current}
        />
      </Container>
    );
  }
}

module.exports = TutorialSidebar;
