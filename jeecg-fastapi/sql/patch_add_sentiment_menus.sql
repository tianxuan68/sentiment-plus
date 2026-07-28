-- 已有库升级：新增评论分析菜单（sa010–sa015）
-- 用法: mysql -u root -p jeecg-boot < sql/patch_add_sentiment_menus.sql

DELETE FROM `sys_role_permission` WHERE `permission_id` IN ('sa010','sa011','sa012','sa013','sa014','sa015');
DELETE FROM `sys_permission` WHERE `id` IN ('sa010','sa011','sa012','sa013','sa014','sa015');

INSERT INTO `sys_permission` (`id`,`parent_id`,`name`,`url`,`component`,`is_route`,`redirect`,`menu_type`,`sort_no`,`icon`,`is_leaf`,`keep_alive`,`hidden`,`del_flag`,`status`) VALUES
('sa010','','评论分析','/sentiment','layouts/RouteView',1,'/sentiment/analyze',0,1.00,'ant-design:comment',0,0,0,0,'1'),
('sa011','sa010','单条分析','/sentiment/analyze','sentiment/analyze/index',1,NULL,1,1.00,'ant-design:edit',1,1,0,0,'1'),
('sa012','sa010','评价看板','/sentiment/dashboard','sentiment/dashboard/index',1,NULL,1,2.00,'ant-design:bar-chart',1,1,0,0,'1'),
('sa013','sa010','优缺点洞察','/sentiment/pros-cons','sentiment/prosCons/index',1,NULL,1,3.00,'ant-design:like',1,1,0,0,'1'),
('sa014','sa010','模型对比','/sentiment/models','sentiment/models/index',1,NULL,1,4.00,'ant-design:fund-outlined',1,1,0,0,'1'),
('sa015','sa010','关键词对比','/sentiment/keywords','sentiment/keywords/index',1,NULL,1,5.00,'ant-design:tags-outlined',1,1,0,0,'1');

UPDATE `sys_permission`
SET `sort_no` = 2.00
WHERE `id` = 'd7d6e2e4e2934f2c9385a623fd98c6f3';

INSERT INTO `sys_role_permission` (`id`,`role_id`,`permission_id`) VALUES
('rp_sa010','f6817f48af4fb3af11b9e8bf182f618b','sa010'),
('rp_sa011','f6817f48af4fb3af11b9e8bf182f618b','sa011'),
('rp_sa012','f6817f48af4fb3af11b9e8bf182f618b','sa012'),
('rp_sa013','f6817f48af4fb3af11b9e8bf182f618b','sa013'),
('rp_sa014','f6817f48af4fb3af11b9e8bf182f618b','sa014'),
('rp_sa015','f6817f48af4fb3af11b9e8bf182f618b','sa015');

SELECT 'added sentiment menus sa010-sa015; default home -> /sentiment/analyze' AS message;
